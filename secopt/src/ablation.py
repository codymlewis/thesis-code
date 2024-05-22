import argparse
import math
import os
from functools import partial

import numpy as np
import jax
import flax.linen as nn
from flax.training import train_state
import einops
import optax
from tqdm import trange
import pandas as pd
import safeflax

import load_datasets
import attack
import common


class CNN(nn.Module):
    classes: int = 10
    activation: str = "relu"
    pooling: str = "none"
    pool_size: str = "small"
    normalisation: str = "none"

    @nn.compact
    def __call__(self, x, representation=False):
        activation_fn = getattr(nn, self.activation)
        if self.pooling == "none":
            def identity(x):
                return x

            pool_fn = identity
        else:
            pool_window = (2, 2) if self.pool_size == "small" else (3, 3)
            pool_fn = partial(getattr(nn, self.pooling), window_shape=pool_window, strides=pool_window)
        if self.normalisation == "none":
            def Identity():
                def _apply(x):
                    return x
                return _apply

            normalisation_fn = Identity
        else:
            normalisation_fn = getattr(nn, self.normalisation)

        x = nn.Conv(48, (3, 3), padding="SAME")(x)
        x = normalisation_fn()(x)
        x = activation_fn(x)
        x = pool_fn(x)
        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = normalisation_fn()(x)
        x = activation_fn(x)
        x = pool_fn(x)
        x = nn.Conv(16, (3, 3), padding="SAME")(x)
        x = normalisation_fn()(x)
        x = activation_fn(x)
        x = pool_fn(x)
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        if representation:
            return x
        x = nn.Dense(self.classes, name="classifier")(x)
        x = nn.softmax(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Ablation study for the impact of the model structure on gradient inversion.")
    parser.add_argument('-a', '--activation', type=str, default="relu", help="Activation function.")
    parser.add_argument('-p', '--pooling', type=str, default="none", help="Type of pooling layers to apply.")
    parser.add_argument('-ps', '--pool-size', type=str, default="small",
                        help="Specify a window size for the pooling layers. [small or large]")
    parser.add_argument('-n', '--normalisation', type=str, default="none", help="Type of normalisation layers to apply.")
    parser.add_argument('-e', '--train_epochs', type=int, default=10, help="The number of epochs to perform training for.")
    parser.add_argument('-d', '--dataset', type=str, default="cifar10", help="Dataset to train on.")
    parser.add_argument('--attack', type=str, default="representation", help="The attack to perform.")
    parser.add_argument('--lr', type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument('-b', '--batch-size', type=int, default=0,
                        help="Batch for the gradient that the attack is performed upon.")
    parser.add_argument('-z', '--zinit', type=str, default="uniform",
                        help="Choose an initialisation fuction for the dummy data [default: uniform].")
    args = parser.parse_args()

    seed = 56
    batch_size = 8
    attack_runs = 100
    net_config = vars(args)
    print(f"Running ablation with {vars(args)}")

    rng = np.random.default_rng(seed)
    dataset = getattr(load_datasets, args.dataset)()
    model = CNN(
        dataset.nclasses,
        activation=args.activation,
        pooling=args.pooling,
        pool_size=args.pool_size,
        normalisation=args.normalisation
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(seed), dataset['train']['X'][:1]),
        tx=optax.sgd(args.lr),
    )
    checkpoint_file = "checkpoints/{}.safetensor".format(
        '-'.join([f'{k}={v}' for k, v in net_config.items() if k not in ['attack', 'batch_size', 'zinit']])
    )

    if os.path.exists(checkpoint_file):
        state = state.replace(params=safeflax.load_load_file(checkpoint_file))
        print(f"Checkpoint loaded from {checkpoint_file}")
    else:
        for e in (pbar := trange(args.train_epochs)):
            idxs = np.array_split(
                rng.permutation(len(dataset['train']['Y'])), math.ceil(len(dataset['train']['Y']) / batch_size)
            )
            loss_sum = 0.0
            for idx in idxs:
                loss, state = common.update_step(state, dataset['train']['X'][idx], dataset['train']['Y'][idx])
                loss_sum += loss
            pbar.set_postfix_str(f"LOSS: {loss_sum / len(idxs):.3f}")
        safeflax.save_file(state.params, checkpoint_file)
        print(f"Checkpoints were saved to {checkpoint_file}")
    final_accuracy = common.accuracy(state, dataset['test']['X'], dataset['test']['Y'], batch_size=batch_size)
    print(f"Final accuracy: {final_accuracy:.3%}")

    all_results = {k: [v for _ in range(attack_runs)] for k, v in net_config.items()}
    all_results["accuracy"] = [final_accuracy for _ in range(attack_runs)]
    all_results["attack"] = [args.attack for _ in range(attack_runs)]
    all_results.update({"seed": [], "psnr": [], "ssim": []})
    for i in range(0, attack_runs):
        attack_seed = round(i**2 + i * np.cos(i * np.pi / 4)) % 2**31
        print(f"Performing the attack with {attack_seed=}, {i=}")
        Z, labels, idx = attack.perform_attack(
            state,
            dataset,
            args.attack,
            {"batch_size": args.batch_size if args.batch_size > 0 else batch_size, "pgd": False},
            seed=attack_seed,
            zinit=args.zinit,
        )
        results = attack.measure_leakage(dataset['train']['X'][idx], Z, dataset['train']['Y'][idx], labels)
        tuned_Z = attack.tune_brightness(Z.copy(), dataset['train']['X'][idx])
        tuned_results = attack.measure_leakage(dataset['train']['X'][idx], tuned_Z, dataset['train']['Y'][idx], labels)
        if np.all([tuned_results[k] > results[k] for k in results.keys()]):
            print("Tuned brightness got better results, so using that")
            Z = tuned_Z
            results = tuned_results
        for k, v in results.items():
            all_results[k].append(v)
        all_results["seed"].append(attack_seed)
        print(f"Attack performance: {results}")

    df = pd.DataFrame(all_results)
    print("Summary of the results")
    print(df.describe())
    os.makedirs("results", exist_ok=True)
    results_fn = "{}{}{}ablation_results.csv".format(
        "rp_" if args.zinit == "repeated_pattern" else "sc_" if args.zinit == "colour" else "",
        f"bs{args.batch_size}_" if args.batch_size > 0 else "",
        f"e{args.train_epochs}_" if args.train_epochs != 10 else ""
    )
    df.to_csv(
        f"results/{results_fn}",
        mode='a',
        header=not os.path.exists(f"results/{results_fn}"),
        index=False,
    )
    print(f"Added results to results/{results_fn}")
