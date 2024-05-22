import argparse
import math
import os
import numpy as np
import jax
from flax.training import train_state
from tqdm import trange
import safeflax

import load_datasets
import models
import common


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generation operations.")
    parser.add_argument('-e', '--epochs', type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument('-b', '--batch-size', type=int, default=8, help="Training and evaluation batch size.")
    parser.add_argument('-d', '--dataset', type=str, default="fmnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="LeNet", help="Neural network model to train.")
    parser.add_argument('-o', '--optimiser', type=str, default="sgd", help="Optimiser to use for training.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help="Learning rate to use for training.")
    parser.add_argument('-p', '--pgd', action="store_true", help="Perform projected gradient descent hardening.")
    parser.add_argument('--perturb', action="store_true", help="Perturb the training data.")
    args = parser.parse_args()

    print(f"Training with {vars(args)}")
    rng = np.random.default_rng(args.seed)
    dataset = getattr(load_datasets, args.dataset)()
    model = getattr(models, args.model)(dataset.nclasses)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(args.seed), dataset['train']['X'][:1]),
        tx=common.find_optimiser(args.optimiser)(args.learning_rate),
    )

    checkpoint_file = "checkpoints/{}.safetensors".format('-'.join([f'{k}={v}' for k, v in vars(args).items()]))
    update_step = common.pgd_update_step if args.pgd else common.update_step

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
    os.makedirs("checkpoints", exist_ok=True)
    safeflax.save_file(state.params, checkpoint_file)
    final_accuracy = common.accuracy(state, dataset['test']['X'], dataset['test']['Y'], batch_size=args.batch_size)
    print(f"Final accuracy: {final_accuracy:.3%}")
    print(f"Checkpoints were saved to {checkpoint_file}")

    accuracy_file = "results/accuracies.csv"
    os.makedirs("results", exist_ok=True)
    training_details = vars(args)
    training_details['accuracy'] = final_accuracy
    if not os.path.exists(accuracy_file):
        with open(accuracy_file, 'w') as f:
            f.write(",".join(training_details.keys()))
            f.write("\n")
    with open(accuracy_file, 'a') as f:
        f.write(','.join([str(v) for v in training_details.values()]))
        f.write("\n")
    print(f"Accuracy details written to {accuracy_file}")
