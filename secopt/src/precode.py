from typing import Optional
import argparse
import math
import os
import gc
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import flax.linen as nn
import optax
import einops
import jaxopt
import sklearn.metrics as skm
import pandas as pd
from tqdm import trange
import safeflax

import load_datasets
import attack
import common
import optimisers


class VBTrainState(train_state.TrainState):
    "Extended train state to keep track of rng used in bottleneck"
    key: jax.Array


class VariationalBottleNeck(nn.Module):
    "The variational bottleneck layer"
    K: int = 256
    rng_collection: str = "bottleneck"

    @nn.compact
    def __call__(self, x, rng: Optional[jax.random.PRNGKey] = None):
        batch_size = x.shape[0]
        in_shape = x.shape[1:]

        if len(in_shape) > 1 and x.shape[1] > 1 and len(in_shape) != 3:
            x = nn.Conv(1, (1, 1), padding="SAME")(x)
        if len(x.shape) > 2:
            x = x.reshape(batch_size, -1)

        statistics = nn.Dense(2 * self.K)(x)
        mu = statistics[:, :self.K]
        std = nn.softplus(statistics[:, self.K:])
        if rng is None:
            rng = self.make_rng(self.rng_collection)
        eps = jax.random.normal(rng, std.shape, dtype=std.dtype)
        encoding = mu + eps * std
        x = nn.Dense(x.shape[1])(encoding)
        x = x.reshape((batch_size, *in_shape))
        return x, mu, std


class MLP(nn.Module):
    "A simple LeNet-300-100 with a variational bottleneck"
    classes: int = 10

    @nn.compact
    def __call__(self, x, training=False, representation=False):
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        if representation:
            return x
        x, mu, std = VariationalBottleNeck(name="bottleneck")(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        if training:
            return x, mu, std
        return x


class CNN(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, training=False, representation=False):
        x = nn.Conv(48, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(16, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        if representation:
            return x
        x, mu, std = VariationalBottleNeck(name="bottleneck")(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        if training:
            return x, mu, std
        return x


class LeNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, training=False, representation=False):
        x = nn.Conv(6, (5, 5), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))
        x = nn.Conv(16, (5, 5), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        if representation:
            return x
        x, mu, std = VariationalBottleNeck(name="bottleneck")(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        if training:
            return x, mu, std
        return x


# ConvNext
class ConvNeXt(nn.Module):
    classes: int

    @nn.compact
    def __call__(self, x, training=False, representation=False):
        depths = [3, 3, 27, 3]
        projection_dims = [128, 256, 512, 1024]
        # Stem block.
        stem = nn.Sequential([
            nn.Conv(projection_dims[0], (4, 4), strides=(4, 4), name="convnext_base_stem_conv"),
            nn.LayerNorm(epsilon=1e-6, name="convnext_base_stem_layernorm"),
        ])

        # Downsampling blocks.
        downsample_layers = [stem]

        num_downsample_layers = 3
        for i in range(num_downsample_layers):
            downsample_layer = nn.Sequential([
                nn.LayerNorm(epsilon=1e-6, name=f"convnext_base_downsampling_layernorm_{i}"),
                nn.Conv(projection_dims[i + 1], (2, 2), strides=(2, 2), name=f"convnext_base_downsampling_conv_{i}"),
            ])
            downsample_layers.append(downsample_layer)

        num_convnext_blocks = 4
        for i in range(num_convnext_blocks):
            x = downsample_layers[i](x)
            for j in range(depths[i]):
                x = ConvNeXtBlock(
                    projection_dim=projection_dims[i],
                    layer_scale_init_value=1e-6,
                    name=f"convnext_base_stage_{i}_block_{j}",
                )(x)

        x = einops.reduce(x, 'b h w c -> b c', 'mean')
        if representation:
            return x
        x, mu, std = VariationalBottleNeck(name="bottleneck")(x)
        x = nn.LayerNorm(epsilon=1e-6, name="convnext_base_head_layernorm")(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        if training:
            return x, mu, std
        return x


class ConvNeXtBlock(nn.Module):
    projection_dim: int
    name: str = None
    layer_scale_init_value: float = 1e-6

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        x = nn.Conv(
            self.projection_dim,
            kernel_size=(7, 7),
            padding="SAME",
            feature_group_count=self.projection_dim,
            name=self.name + "_depthwise_conv",
        )(x)
        x = nn.LayerNorm(epsilon=1e-6, name=self.name + "_layernorm")(x)
        x = nn.Dense(4 * self.projection_dim, name=self.name + "_pointwise_conv_1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.projection_dim, name=self.name + "_pointwise_conv_2")(x)

        if self.layer_scale_init_value is not None:
            x = LayerScale(
                self.layer_scale_init_value,
                self.projection_dim,
                name=self.name + "_layer_scale",
            )(x)

        return inputs + x


class LayerScale(nn.Module):
    init_values: float
    projection_dim: int
    name: str
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        gamma = self.param(
            'gamma',
            nn.initializers.constant(self.init_values, dtype=self.param_dtype),
            (self.projection_dim,),
            self.param_dtype,
        )
        return x * gamma


# ResNetV2
class ResNetV2(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x, training=False, representation=False):
        x = jnp.pad(x, ((0, 0), (3, 3), (3, 3), (0, 0)))
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="VALID", use_bias=True, name="conv1_conv")(x)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = Stack2(64, 3, name="conv2")(x)
        x = Stack2(128, 4, name="conv3")(x)
        x = Stack2(256, 6, name="conv4")(x)
        x = Stack2(512, 3, strides1=(1, 1), name="conv5")(x)

        x = nn.LayerNorm(epsilon=1.001e-5)(x)
        x = nn.relu(x)

        x = einops.reduce(x, "b h w d -> b d", 'mean')
        if representation:
            return x

        x, mu, std = VariationalBottleNeck(name="bottleneck")(x)
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        if training:
            return x, mu, std
        return x


class Block2(nn.Module):
    filters: int
    kernel: (int, int) = (3, 3)
    strides: (int, int) = (1, 1)
    conv_shortcut: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        preact = nn.LayerNorm(epsilon=1.001e-5)(x)
        preact = nn.relu(preact)

        if self.conv_shortcut:
            shortcut = nn.Conv(
                4 * self.filters, (1, 1), strides=self.strides, padding="VALID", name=self.name + "_0_conv"
            )(preact)
        else:
            shortcut = nn.max_pool(x, (1, 1), strides=self.strides) if self.strides > (1, 1) else x

        x = nn.Conv(
            self.filters, (1, 1), strides=(1, 1), padding="VALID", use_bias=False,
            name=self.name + "_1_conv"
        )(preact)
        x = nn.LayerNorm(epsilon=1.001e-5)(x)
        x = nn.relu(x)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        x = nn.Conv(
            self.filters, self.kernel, strides=self.strides, padding="VALID", use_bias=False,
            name=self.name + "_2_conv"
        )(x)
        x = nn.LayerNorm(epsilon=1.001e-5)(x)
        x = nn.relu(x)

        x = nn.Conv(4 * self.filters, (1, 1), name=self.name + "_3_conv")(x)
        x = shortcut + x
        return x


class Stack2(nn.Module):
    filters: int
    blocks: int
    strides1: (int, int) = (2, 2)
    name: str = None

    @nn.compact
    def __call__(self, x):
        x = Block2(self.filters, conv_shortcut=True, name=self.name + "_block1")(x)
        for i in range(2, self.blocks):
            x = Block2(self.filters, name=f"{self.name}_block{i}")(x)
        x = Block2(self.filters, strides=self.strides1, name=f"{self.name}_block{self.blocks}")(x)
        return x


def get_model(model_name):
    match model_name:
        case "MLP":
            return MLP
        case "CNN":
            return CNN
        case "LeNet":
            return LeNet
        case "ResNetV2":
            return ResNetV2
        case "ConvNeXt":
            return ConvNeXt
        case _:
            raise AttributeError(f"Model {args.model} not implemented.")


@jax.jit
def update_step(state, X, Y, beta: float = 1e-3):
    "Function for a step of training the model"
    vb_train_key = jax.random.fold_in(key=state.key, data=state.step)

    def loss_fn(params):
        logits, mu, std = state.apply_fn(params, X, training=True, rngs={'bottleneck': vb_train_key})
        logits = jnp.clip(logits, 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        ce_loss = -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
        vb_loss = -0.5 * einops.reduce(1 + 2 * jnp.log(std) - mu**2 - std**2, 'b i -> b', 'sum').mean() / jnp.log(2)
        return ce_loss + beta * vb_loss

    state = state.replace(key=vb_train_key)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


@jax.jit
def pgd_update_step(state, X, Y, epsilon=4/255, pgd_steps=3, beta: float = 1e-3):
    vb_train_key = jax.random.fold_in(key=state.key, data=state.step)

    def loss_fn(params, dX):
        logits, mu, std = state.apply_fn(params, X, training=True, rngs={'bottleneck': vb_train_key})
        logits = jnp.clip(logits, 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        ce_loss = -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
        vb_loss = -0.5 * einops.reduce(1 + 2 * jnp.log(std) - mu**2 - std**2, 'b i -> b', 'sum').mean() / jnp.log(2)
        return ce_loss + beta * vb_loss

    pgd_lr = (2 * epsilon) / 3
    X_nat = X
    for _ in range(pgd_steps):
        Xgrads = jax.grad(loss_fn, argnums=1)(state.params, X)
        X = X + pgd_lr * jnp.sign(Xgrads)
        X = jnp.clip(X, X_nat - epsilon, X_nat + epsilon)
        X = jnp.clip(X, 0, 1)

    loss, grads = jax.value_and_grad(loss_fn)(state.params, X)
    state = state.apply_gradients(grads=grads)
    return loss, state


def measure(state, X, Y, batch_size=1000, metric_name="accuracy_score"):
    """
    Measure some metric of the model across the given dataset

    Arguments:
    - state: Flax train state
    - X: The samples
    - Y: The corresponding labels for the samples
    - batch_size: Amount of samples to compute the accuracy on at a time
    - metric from scikit learn metrics to use
    """
    @jax.jit
    def _apply(batch_X, rng_key):
        return state.apply_fn(state.params, batch_X, rngs={'bottleneck': rng_key})

    rng_key = state.key
    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        rng_key = jax.random.fold_in(rng_key, data=i)
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end], rng_key))
        Ys.append(Y[i:i_end])
    if isinstance(metric_name, list):
        return [
            getattr(skm, mn)(
                jnp.concatenate(Ys), jnp.concatenate(preds) if "loss" in mn else jnp.argmax(jnp.concatenate(preds), axis=-1))
            for mn in metric_name
        ]
    return getattr(skm, metric_name)(jnp.concatenate(Ys), jnp.concatenate(preds))


def representation_loss(state, true_reps, l1_reg=0.0, l2_reg=1e-6):
    """
    Representation inversion attack proposed in https://arxiv.org/abs/2202.10546
    """
    def _apply(Z):
        dist = attack.cosine_dist(
            state.apply_fn(state.params, Z, representation=True, rngs={'bottleneck': state.key}),
            true_reps,
        )
        tv_l1, tv_l2 = attack.total_variation(Z)
        return dist + l1_reg * tv_l1 + l2_reg * tv_l2
    return _apply


def perform_attack(
    state,
    dataset,
    train_args,
    seed=42,
    batch_size=None,
    l1_reg=0.0,
    l2_reg=1e-6,
    nsteps=1000,
):
    if batch_size is None:
        batch_size = int(train_args['batch_size'])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset['train']['Y']), batch_size)
    loss, new_state = (pgd_update_step if train_args["pgd"] else update_step)(
        state,
        dataset['train']['X'][idx],
        dataset['train']['Y'][idx],
    )
    true_grads = jax.tree_util.tree_map(lambda a, b: a - b, state.params, new_state.params)
    labels = jnp.argsort(jnp.min(true_grads['params']['classifier']['kernel'], axis=0))[:batch_size]
    true_reps = true_grads['params']['classifier']['kernel'].T[labels]
    solver = jaxopt.OptaxSolver(
        representation_loss(state, true_reps, l1_reg=l1_reg, l2_reg=l2_reg),
        optax.lion(optax.cosine_decay_schedule(0.1, nsteps))
    )

    Z = []
    for label in labels:
        test_idx = np.arange(len(dataset['test']['Y']))[dataset['test']['Y'] == label]
        Z.append(dataset['test']['X'][rng.choice(test_idx, 1)])
    Z = jnp.concatenate(Z, axis=0)

    attack_state = solver.init_state(Z)
    trainer = jax.jit(solver.update)
    for s in (pbar := trange(nsteps)):
        Z, attack_state = trainer(Z, attack_state)
        pbar.set_postfix_str(f"LOSS: {attack_state.value:.5f}")
    Z = jnp.clip(Z, 0, 1)
    Z, labels = np.array(Z), np.array(labels)

    del attack_state
    del trainer
    del true_grads
    return Z, labels, idx


def accuracy(state, X, Y, batch_size=1000):
    "Calculate the accuracy of the model across the given dataset"
    @jax.jit
    def _apply(batch_X, rng_key):
        return jnp.argmax(state.apply_fn(state.params, batch_X, rngs={'bottleneck': rng_key}), axis=-1)

    rng_key = state.key
    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        rng_key = jax.random.fold_in(rng_key, data=i)
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end], rng_key))
        Ys.append(Y[i:i_end])
    return skm.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generation operations.")
    parser.add_argument('-r', '--rounds', type=int, default=3000, help="Number of rounds to train for.")
    parser.add_argument('-e', '--epochs', type=int, default=1, help="Number of epochs to train for each round.")
    parser.add_argument('-st', '--steps', type=int, default=1, help="Number of steps to train for each epoch.")
    parser.add_argument('-c', '--clients', type=int, default=10, help="Number of clients to train.")
    parser.add_argument('-b', '--batch-size', type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument('-d', '--dataset', type=str, default="fmnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="LeNet", help="Neural network model to train.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001,
                        help="Learning rate to use for training.")
    parser.add_argument("--secadam", action="store_true", help="Whether to use SecAdam.")
    parser.add_argument('--pgd', action="store_true", help="Perform projected gradient descent hardening.")
    parser.add_argument('--perturb', action="store_true", help="Perturb the training data.")
    parser.add_argument("--train-inversion", action="store_true", help="Train a model to be attacked")
    parser.add_argument("--performance", action="store_true", help="Evaluate the federated learning performance.")
    parser.add_argument("--attack", action="store_true", help="Perform an attack on a trained model.")
    parser.add_argument('-f', '--file', type=str, help="File containing the checkpoint of the model")
    parser.add_argument('--runs', type=int, default=1, help="Number of runs of the attack to perform.")
    parser.add_argument('--l1-reg', type=float, default=0.0, help="Influence of L1 total variation in the attack")
    parser.add_argument('--l2-reg', type=float, default=1e-6, help="Influence of L2 total variation in the attack")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset = getattr(load_datasets, args.dataset)()
    model = get_model(args.model)(classes=dataset.nclasses)
    params_key, vb_key = jax.random.split(jax.random.PRNGKey(args.seed))
    state = VBTrainState.create(
        apply_fn=model.apply,
        params=model.init(params_key, dataset['train']['X'][:1]),
        tx=optimisers.secadam(args.learning_rate) if args.secadam else optax.sgd,
        key=vb_key,
    )
    update_step = pgd_update_step if args.pgd else update_step
    training_details = vars(args)
    print(f"Performing experiment with {training_details}")

    if args.train_inversion:
        checkpoint_file = "precode_checkpoints/" + \
            "dataset={}-seed={}-learning_rate={}-pgd={}-model={}-perturb={}-batch_size={}-secadam={}.safetensors".format(
                args.dataset,
                args.seed,
                args.learning_rate,
                args.pgd,
                args.model,
                args.perturb,
                args.batch_size,
                args.secadam,
            )

        for e in (pbar := trange(args.epochs)):
            if args.perturb:
                dataset.perturb(rng)
            idxs = np.array_split(
                rng.permutation(len(dataset['train']['Y'])), math.ceil(len(dataset['train']['Y']) / args.batch_size)
            )
            loss_sum = 0.0
            for idx in idxs:
                loss, state = update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])
                loss_sum += loss
            pbar.set_postfix_str(f"LOSS: {loss_sum / len(idxs):.3f}")
        os.makedirs("precode_checkpoints", exist_ok=True)
        safeflax.save_file(state.params, checkpoint_file)
        final_accuracy = accuracy(state, dataset['test']['X'], dataset['test']['Y'], batch_size=args.batch_size)
        print(f"Final accuracy: {final_accuracy:.3%}")
        print(f"Checkpoints were saved to {checkpoint_file}")

        results_file = "results/precode_accuracies.csv"
        training_details['accuracy'] = final_accuracy

    if args.performance:
        global_state = state.replace(tx=optax.sgd(args.learning_rate))
        client_keys = iter(jax.random.split(params_key, args.clients))
        client_states = [
                VBTrainState.create(
                    apply_fn=model.apply,
                    params=global_state.params,
                    tx=optimisers.secadam(args.learning_rate) if args.secadam else optax.sgd(args.learning_rate),
                    key=next(client_keys),
                )
                for _ in range(args.clients)
            ]
        idxs = common.lda(dataset['train']['Y'], args.clients, dataset.nclasses, rng, alpha=0.5)
        client_data = [
            {"X": dataset['train']['X'][idx], "Y": dataset['train']['Y'][idx]} for idx in idxs
        ]

        for _ in (pbar := trange(args.rounds)):
            full_loss_sum = 0.0
            if args.secadam:
                all_mus, all_nus = [], []
            else:
                all_updates = []
            for c in range(args.clients):
                client_states[c] = client_states[c].replace(params=global_state.params)
                for _ in range(args.epochs):
                    client_data_len = len(client_data[c]['Y'])
                    if args.steps > 0:
                        idxs = rng.choice(
                            client_data_len,
                            (args.steps, args.batch_size),
                            replace=client_data_len >= args.steps * args.batch_size
                        )
                    else:
                        idxs = np.array_split(
                            rng.permutation(len(client_data[c]['Y'])), math.ceil(len(client_data[c]['Y']) / args.batch_size)
                        )
                    loss_sum = 0.0
                    for idx in idxs:
                        loss, client_states[c] = update_step(
                            client_states[c],
                            client_data[c]['X'][idx],
                            client_data[c]['Y'][idx],
                        )
                        loss_sum += loss
                full_loss_sum += loss_sum / len(idxs)

                if args.secadam:
                    mu, nu = common.find_secadam_update(client_states[c])
                    all_mus.append(mu)
                    all_nus.append(nu)
                else:
                    all_updates.append(common.find_update(global_state, client_states[c], args.learning_rate))
            if args.secadam:
                global_grads = common.secadam_agg(all_mus, all_nus)
            else:
                global_grads = common.fedavg(all_updates)
            global_state = global_state.apply_gradients(grads=global_grads)
            pbar.set_postfix_str(f"LOSS: {full_loss_sum / args.clients:.3f}")
        final_accuracy, final_loss = measure(
            global_state,
            dataset['test']['X'],
            dataset['test']['Y'],
            batch_size=args.batch_size,
            metric_name=["accuracy_score", "log_loss"],
        )
        print(f"Final accuracy: {final_accuracy:.3%}, Final Loss: {final_loss:.5f}")

        results_file = "results/precode_performance_results.csv"
        training_details['accuracy'] = final_accuracy
        training_details['loss'] = final_loss

    os.makedirs('results', exist_ok=True)
    if args.train_inversion or args.performance:
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                f.write(",".join(training_details.keys()))
                f.write("\n")
        with open(results_file, 'a') as f:
            f.write(','.join([str(v) for v in training_details.values()]))
            f.write("\n")
        print(f"Results written to {results_file}")

    if args.attack:
        train_args = {
            a.split('=')[0]: a.split('=')[1]
            for a in args.file.replace(".safetensors", "")[args.file.rfind('/') + 1:].split('-')
        }
        dataset = getattr(load_datasets, train_args['dataset'])()
        model = get_model(train_args["model"])(dataset.nclasses)
        if train_args['perturb'] == "True":
            dataset.perturb(np.random.default_rng(int(train_args['seed']) + 1))
        init_params = safeflax.load_file(args.file)
        learning_rate = float(train_args["learning_rate"])
        opt = optimisers.secadam(learning_rate) if train_args["secadam"] == "True" else optax.sgd(learning_rate)
        state = VBTrainState.create(
            apply_fn=model.apply,
            params=init_params,
            tx=opt,
            key=vb_key,
        )

        all_results = {
            k: [v for _ in range(args.runs)]
            for k, v in train_args.items() if k in ["dataset", "model", "pgd"]
        }
        all_results['attack'] = [args.attack for _ in range(args.runs)]
        all_results['batch_size'] = [args.batch_size for _ in range(args.runs)]
        all_results['l1_reg'] = [args.l1_reg for _ in range(args.runs)]
        all_results['l2_reg'] = [args.l2_reg for _ in range(args.runs)]
        all_results.update({"seed": [], "psnr": [], "ssim": []})

        for i in range(0, args.runs):
            seed = round(i**2 + i * np.cos(i * np.pi / 4)) % 2**31
            print(f"Performing the attack with {seed=}")
            Z, labels, idx = perform_attack(
                state,
                dataset,
                train_args,
                seed=seed,
                batch_size=args.batch_size,
                l1_reg=args.l1_reg,
                l2_reg=args.l2_reg,
            )
            results = attack.measure_leakage(dataset['train']['X'][idx], Z, dataset['train']['Y'][idx], labels)
            tuned_Z = attack.tune_brightness(Z.copy(), dataset['train']['X'][idx])
            tuned_results = attack.measure_leakage(
                dataset['train']['X'][idx], tuned_Z, dataset['train']['Y'][idx], labels
            )
            if np.all([tuned_results[k] > results[k] for k in results.keys()]):
                print("Tuned brightness got better results, so using that")
                Z = tuned_Z
                results = tuned_results
            for k, v in results.items():
                all_results[k].append(v)
            all_results["seed"].append(seed)
            print(f"Attack performance: {results}")
            gc.collect()
        full_results = pd.DataFrame.from_dict(all_results)
        print("Summary results:")
        print(full_results.describe())
        results_fn = "results/precode_inversion_results.csv"
        full_results.to_csv(results_fn, mode='a', header=not os.path.exists(results_fn), index=False)
        print(f"Added results to {results_fn}")
