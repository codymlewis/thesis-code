import argparse
import math
import os
import gc

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import jaxopt
from tqdm import trange
import matplotlib.pyplot as plt
import skimage as ski
import skimage.metrics as skim
import pandas as pd
import safeflax

import models
import load_datasets
import common


def idlg_loss(state, update_step, true_grads, labels):
    "https://arxiv.org/abs/2001.02610"
    def _apply(Z):
        _, new_state = update_step(state, Z, labels)
        new_grads = jax.tree_util.tree_map(lambda a, b: a - b, state.params, new_state.params)
        norm_tree = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b)**2), new_grads, true_grads)
        return jnp.sqrt(jax.tree_util.tree_reduce(lambda a, b: a + b, norm_tree))
    return _apply


def cpl_loss(state, update_step, true_grads, labels, alpha=0.5):
    "https://link.springer.com/chapter/10.1007/978-3-030-58951-6_27"
    def _apply(Z):
        _, new_state = update_step(state, Z, labels)
        new_grads = jax.tree_util.tree_map(lambda a, b: a - b, state.params, new_state.params)
        norm_tree = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b)**2), new_grads, true_grads)

        logits = jnp.clip(state.apply_fn(state.params, Z), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        y_pred_dist = jnp.linalg.norm(logits - one_hot)
        return jnp.sqrt(jax.tree_util.tree_reduce(lambda a, b: a + b, norm_tree)) + alpha * y_pred_dist
    return _apply


def reg_idlg_loss(state, update_step, true_grads, labels, alpha=0.5, lamb_tv=1e-6):
    "iDLG attack with regularisation"
    def _apply(Z):
        _, new_state = update_step(state, Z, labels)
        new_grads = jax.tree_util.tree_map(lambda a, b: a - b, state.params, new_state.params)
        norm_tree = jax.tree_util.tree_map(lambda a, b: jnp.sum((a - b)**2), new_grads, true_grads)

        logits = jnp.clip(state.apply_fn(state.params, Z), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        return jnp.sqrt(jax.tree_util.tree_reduce(lambda a, b: a + b, norm_tree)) + \
            alpha * (cosine_dist(logits, one_hot) + lamb_tv * total_variation(Z))
    return _apply


def cosine_dist(A, B):
    denom = jnp.maximum(jnp.linalg.norm(A, axis=-1) * jnp.linalg.norm(B, axis=-1), 1e-15)
    return 1 - jnp.mean(jnp.abs(jnp.einsum('br,br -> b', A, B)) / denom)


def total_variation(V):
    pixel_dif1 = V[:, 1:, :] - V[:, :-1, :]
    pixel_dif2 = V[:, :, 1:] - V[:, :, :-1]
    tv_l1 = jnp.abs(pixel_dif1).sum() + jnp.abs(pixel_dif2).sum()
    tv_l2 = (pixel_dif1**2).sum() + (pixel_dif2**2).sum()
    return tv_l1, tv_l2


def representation_loss(state, true_reps, l1_reg=0.0, l2_reg=1e-6):
    """
    Representation inversion attack proposed in https://arxiv.org/abs/2202.10546
    """
    def _apply(Z):
        dist = cosine_dist(state.apply_fn(state.params, Z, representation=True), true_reps)
        tv_l1, tv_l2 = total_variation(Z)
        return dist + l1_reg * tv_l1 + l2_reg * tv_l2
    return _apply


def plot_image_array(images, filename):
    # Plot a grid of images
    batch_size = len(images)
    if batch_size > 1:
        if batch_size > 3:
            nrows = math.floor(math.sqrt(batch_size))
            ncols = batch_size // nrows
        else:
            nrows, ncols = 1, batch_size
        fig, axes = plt.subplots(nrows, ncols)
        for i, ax in enumerate(axes.flatten()):
            if i < batch_size:
                ax.imshow(images[i], cmap='binary')
            ax.axis('off')
    else:
        plt.imshow(images[0], cmap="binary")
    plt.tight_layout()
    plt.savefig(filename, dpi=320)
    plt.clf()


def measure_leakage(true_X, Z, true_Y, labels):
    if true_X.shape[-1] > 1:
        true_X = np.array([ski.color.rgb2gray(x) for x in true_X]).reshape(true_X.shape[:-1] + (1,))
    if Z.shape[-1] > 1:
        Z = np.array([ski.color.rgb2gray(z) for z in Z]).reshape(Z.shape[:-1] + (1,))
    metrics = {"ssim": [], "psnr": []}
    for tx in true_X:
        for z in Z:
            metrics['ssim'].append(skim.structural_similarity(tx, z, channel_axis=2, data_range=1.0))
            metrics['psnr'].append(skim.peak_signal_noise_ratio(tx, z, data_range=1.0))
    return {"ssim": max(metrics['ssim']), "psnr": max(metrics['psnr'])}


def perform_attack(
    state,
    dataset,
    attack,
    train_args,
    seed=42,
    steps=1,
    zinit="uniform",
    l1_reg=0.0,
    l2_reg=1e-6,
    nsteps=1000,
    client_regularisation=False,
):
    batch_size = int(train_args['batch_size'])
    rng = np.random.default_rng(seed)
    update_step = common.pgd_update_step if train_args["pgd"] else common.update_step
    new_state = state
    for step in range(steps):
        idx = rng.choice(len(dataset['train']['Y']), batch_size)
        _, new_state = update_step(
            new_state,
            dataset['train']['X'][idx],
            dataset['train']['Y'][idx],
            lamb=0.01 if client_regularisation else 0.0,
        )

    true_grads = jax.tree_util.tree_map(lambda a, b: a - b, state.params, new_state.params)
    labels = jnp.argsort(jnp.min(true_grads['params']['classifier']['kernel'], axis=0))[:batch_size]

    match attack:
        case "representation":
            true_reps = true_grads['params']['classifier']['kernel'].T[labels]
            solver = jaxopt.OptaxSolver(
                representation_loss(state, true_reps, l1_reg=l1_reg, l2_reg=l2_reg),
                optax.lion(optax.cosine_decay_schedule(0.1, nsteps))
            )
        case "cpl":
            solver = jaxopt.LBFGS(cpl_loss(state, update_step, true_grads, labels))
        case "idlg":
            solver = jaxopt.LBFGS(idlg_loss(state, update_step, true_grads, labels))
        case "reg_idlg":
            solver = jaxopt.LBFGS(reg_idlg_loss(state, update_step, true_grads, labels))
        case _:
            raise NotImplementedError(f"Attack {attack} is not implemented.")

    match zinit:
        case "uniform":
            Z = jax.random.uniform(jax.random.PRNGKey(seed), shape=(batch_size,) + dataset.input_shape)
        case "normal":
            Z = jax.random.normal(jax.random.PRNGKey(seed), shape=(batch_size,) + dataset.input_shape) * 0.2 + 0.5
        case "data":
            Z = []
            for label in labels:
                test_idx = np.arange(len(dataset['test']['Y']))[dataset['test']['Y'] == label]
                Z.append(dataset['test']['X'][rng.choice(test_idx, 1)])
            Z = jnp.concatenate(Z, axis=0)
        case "repeated_pattern":
            pattern = jax.random.uniform(
                jax.random.PRNGKey(seed),
                shape=(batch_size,) + tuple(i // 8 for i in dataset.input_shape[:-1]) + (dataset.input_shape[-1],)
            )
            Z = jnp.repeat(pattern, 8, axis=1)
            Z = jnp.repeat(Z, 8, axis=2)
        case "colour":
            channel_data = jnp.ones((batch_size,) + dataset.input_shape[:-1] + (1,), dtype=jnp.float32)
            if dataset.input_shape[-1] > 1:
                other_channels = jnp.zeros(
                    (batch_size,) + dataset.input_shape[:-1] + (dataset.input_shape[-1] - 1,),
                    dtype=jnp.float32
                )
                channel_data = jnp.concatenate((channel_data, other_channels), axis=-1)
            Z = channel_data
        case _:
            raise NotImplementedError(f"Dummy data initialisation {zinit} is not implemented.")

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


def tune_brightness(Z, ground_truth):
    "Tune the brightness of the recreated images with ground truth"
    Z *= ground_truth.std() / Z.std()
    Z += ground_truth.mean() + Z.mean()
    Z = np.clip(Z, 0, 1)
    return Z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-f', '--file', type=str, required=True, help="File containing the checkpoint of the model")
    parser.add_argument('-r', '--runs', type=int, default=1, help="Number of runs of the attack to perform.")
    parser.add_argument('-p', '--plot', action="store_true", help="Whether to plot the final results.")
    parser.add_argument('-a', '--attack', type=str, default="representation",
                        help="The type of gradient inversion attack to perform.")
    parser.add_argument('-b', '--batch-size', type=int, default=0, help="Override the batch size used in training.")
    parser.add_argument('-o', '--optimiser', type=str, default=None, help="Override the optimiser used in training.")
    parser.add_argument('-z', '--zinit', type=str, default="uniform",
                        help="Choose an initialisation fuction for the dummy data [default: uniform].")
    parser.add_argument('--l1-reg', type=float, default=0.0, help="Influence of L1 total variation in the attack")
    parser.add_argument('--l2-reg', type=float, default=1e-6, help="Influence of L2 total variation in the attack")
    parser.add_argument('--clip-threshold', type=float, default=10.0,
                        help="Clip gradients to this maximum norm if DP optimisation")
    parser.add_argument('--noise-scale', type=float, default=0.00001,
                        help="Scale of noise applied to gradient if DP optimisation")
    parser.add_argument('-reg', '--regularise', action='store_true', help="Apply L2 regularisation to training.")
    parser.add_argument('--steps', type=int, default=1,
                        help="Number of steps of training before an attack is performed.")
    args = parser.parse_args()

    train_args = {
        a.split('=')[0]: a.split('=')[1] for a in args.file.replace(".safetensors", "")[args.file.rfind('/') + 1:].split('-')
    }
    print(f"Performing the attack with {vars(args)}")
    dataset = getattr(load_datasets, train_args['dataset'])()
    if train_args['perturb'] == "True":
        dataset.perturb(np.random.default_rng(int(train_args['seed']) + 1))
    model = getattr(models, train_args["model"])(dataset.nclasses)
    if args.attack == "representation":
        init_params = safeflax.load_file(args.file)
    else:
        init_params = model.init(jax.random.PRNGKey(train_args['seed'], dataset['train'][:1]))
    if args.optimiser is not None:
        if args.optimiser.startswith("dp"):
            opt = common.find_optimiser(args.optimiser)(
                float(train_args["learning_rate"]), clip_threshold=args.clip_threshold, noise_scale=args.noise_scale)
        else:
            opt = common.find_optimiser(args.optimiser)(float(train_args["learning_rate"]))
    else:
        opt = common.find_optimiser(train_args["optimiser"])(float(train_args["learning_rate"]))
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        tx=opt,
    )

    if args.batch_size:
        train_args['batch_size'] = args.batch_size

    all_results = {
        k: [v for _ in range(args.runs)]
        for k, v in train_args.items() if k in ["dataset", "model", "pgd"]
    }
    all_results['attack'] = [args.attack for _ in range(args.runs)]
    if args.optimiser:
        if args.optimiser.startswith("dp"):
            opt_name = f"{args.optimiser}_ct{args.clip_threshold}_ns{args.noise_scale}"
        else:
            opt_name = args.optimiser
    else:
        opt_name = train_args["optimiser"]
    all_results['optimiser'] = [opt_name for _ in range(args.runs)]
    all_results['batch_size'] = [args.batch_size for _ in range(args.runs)]
    all_results['steps'] = [args.steps for _ in range(args.runs)]
    all_results['regularise'] = [args.regularise for _ in range(args.runs)]
    all_results['l1_reg'] = [args.l1_reg for _ in range(args.runs)]
    all_results['l2_reg'] = [args.l2_reg for _ in range(args.runs)]
    all_results.update({"seed": [], "psnr": [], "ssim": []})
    if args.plot:
        all_Zs, all_idxs = [], []
    for i in range(0, args.runs):
        seed = round(i**2 + i * np.cos(i * np.pi / 4)) % 2**31
        print(f"Performing the attack with {seed=}")
        Z, labels, idx = perform_attack(
            state,
            dataset,
            args.attack,
            train_args,
            seed=seed,
            steps=args.steps,
            zinit=args.zinit,
            l1_reg=args.l1_reg,
            l2_reg=args.l2_reg,
            client_regularisation=args.regularise,
        )
        results = measure_leakage(dataset['train']['X'][idx], Z, dataset['train']['Y'][idx], labels)
        tuned_Z = tune_brightness(Z.copy(), dataset['train']['X'][idx])
        tuned_results = measure_leakage(dataset['train']['X'][idx], tuned_Z, dataset['train']['Y'][idx], labels)
        if np.all([tuned_results[k] > results[k] for k in results.keys()]):
            print("Tuned brightness got better results, so using that")
            Z = tuned_Z
            results = tuned_results
        for k, v in results.items():
            all_results[k].append(v)
        all_results["seed"].append(seed)
        print(f"Attack performance: {results}")
        if args.plot:
            all_Zs.append(Z)
            all_idxs.append(idx)
        gc.collect()
    if args.plot:
        Z = jnp.concatenate(all_Zs, axis=0)
        idx = jnp.concatenate(all_idxs, axis=0)
        os.makedirs("plots", exist_ok=True)
        print("Making a plot of the ground truth")
        plot_image_array(
            dataset['train']['X'][idx], f"plots/{train_args['dataset']}_ground_truth.png"
        )
        print("Making a plot of the attack images")
        plot_image_array(
            Z,
            "plots/{}_{}_{}_{}{}.png".format(
                args.attack,
                train_args['model'],
                train_args['dataset'],
                args.optimiser if args.optimiser else train_args['optimiser'],
                '_pgd' if train_args['pgd'] == 'True' else ''
            )
        )
    full_results = pd.DataFrame.from_dict(all_results)
    print("Summary results:")
    print(full_results.describe())
    os.makedirs("results", exist_ok=True)
    results_fn = "results/inversion_results.csv"
    full_results.to_csv(results_fn, mode='a', header=not os.path.exists(results_fn), index=False)
    print(f"Added results to {results_fn}")
