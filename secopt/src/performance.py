import argparse
import functools
import math
import os
import numpy as np
import jax
from flax.training import train_state
from tqdm import trange

import load_datasets
import models
import common


def find_optimiser(opt_name, clip_threshold, noise_scale):
    if opt_name.startswith("dp"):
        return functools.partial(common.find_optimiser(opt_name), clip_threshold=clip_threshold, noise_scale=noise_scale)
    return common.find_optimiser(opt_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network models for inversion attacks.")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generation operations.")
    parser.add_argument('-c', '--clients', type=int, default=10, help="Number of clients to train.")
    parser.add_argument('-r', '--rounds', type=int, default=3000, help="Number of rounds to train for.")
    parser.add_argument('-e', '--epochs', type=int, default=1, help="Number of epochs to train for each round.")
    parser.add_argument('-st', '--steps', type=int, default=1, help="Number of steps to train for each epoch.")
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="Training and evaluation batch size.")
    parser.add_argument('-d', '--dataset', type=str, default="fmnist", help="Dataset to train on.")
    parser.add_argument('-m', '--model', type=str, default="LeNet", help="Neural network model to train.")
    parser.add_argument('-so', '--server-optimiser', type=str, default="sgd",
                        help="Optimiser for the server to use for training.")
    parser.add_argument('-co', '--client-optimiser', type=str, default="sgd",
                        help="Optimiser for the clients to use for training.")
    parser.add_argument('-slr', '--server-learning-rate', type=float, default=0.001,
                        help="Learning rate for the server to use for training.")
    parser.add_argument('-clr', '--client-learning-rate', type=float, default=0.001,
                        help="Learning rate for the clients to use for training.")
    parser.add_argument('-p', '--pgd', action="store_true", help="Perform projected gradient descent hardening.")
    parser.add_argument('-pr', '--participation-rate', type=float, default=1.0,
                        help="Participation rate of clients in each round.")
    parser.add_argument('-ct', '--clip-threshold', type=float, default=1.0,
                        help="Clip gradients to this maximum norm if DP optimisation")
    parser.add_argument('-ns', '--noise-scale', type=float, default=0.1,
                        help="Scale of noise applied to gradient if DP optimisation")
    parser.add_argument('-reg', '--regularise', action='store_true', help="Apply L2 regularisation to training.")
    args = parser.parse_args()

    print(f"Training with {vars(args)}")
    rng = np.random.default_rng(args.seed)
    dataset = getattr(load_datasets, args.dataset)()
    model = getattr(models, args.model)(dataset.nclasses)

    global_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(args.seed), dataset['train']['X'][:1]),
        tx=find_optimiser(args.server_optimiser, args.clip_threshold, args.noise_scale)(args.server_learning_rate),
    )
    client_states = [
            train_state.TrainState.create(
                apply_fn=model.apply,
                params=model.init(jax.random.PRNGKey(args.seed), dataset['train']['X'][:1]),
                tx=find_optimiser(args.client_optimiser, args.clip_threshold, args.noise_scale)(args.client_learning_rate),
            )
            for _ in range(args.clients)
        ]
    idxs = common.lda(dataset['train']['Y'], args.clients, dataset.nclasses, rng, alpha=0.5)
    client_data = [
        {"X": dataset['train']['X'][idx], "Y": dataset['train']['Y'][idx]} for idx in idxs
    ]
    update_step = common.pgd_update_step if args.pgd else common.update_step

    for _ in (pbar := trange(args.rounds)):
        full_loss_sum = 0.0
        if args.client_optimiser == "secadam":
            all_mus, all_nus = [], []
        else:
            all_updates = []
        chosen_clients = rng.choice(args.clients, max(1, round(args.participation_rate * args.clients)), replace=False)
        for c in chosen_clients:
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
                        lamb=0.01 if args.regularise else 0.0,
                    )
                    loss_sum += loss
            full_loss_sum += loss_sum / len(idxs)

            if args.client_optimiser == "secadam":
                mu, nu = common.find_secadam_update(client_states[c])
                all_mus.append(mu)
                all_nus.append(nu)
            else:
                all_updates.append(common.find_update(global_state, client_states[c], args.client_learning_rate))
        if args.client_optimiser == "secadam":
            global_grads = common.secadam_agg(all_mus, all_nus)
        else:
            global_grads = common.fedavg(all_updates)
        global_state = global_state.apply_gradients(grads=global_grads)
        pbar.set_postfix_str(f"LOSS: {full_loss_sum / args.clients:.3f}")
    final_accuracy, final_loss = common.measure(
        global_state,
        dataset['test']['X'],
        dataset['test']['Y'],
        batch_size=args.batch_size,
        metric_name=["accuracy_score", "log_loss"],
    )
    print(f"Final accuracy: {final_accuracy:.3%}, Final Loss: {final_loss:.5f}")

    os.makedirs('results', exist_ok=True)
    results_file = "results/performance_results.csv"
    training_details = vars(args)
    training_details['accuracy'] = final_accuracy
    training_details['loss'] = final_loss
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write(",".join(training_details.keys()))
            f.write("\n")
    with open(results_file, 'a') as f:
        f.write(','.join([str(v) for v in training_details.values()]))
        f.write("\n")
    print(f"Results written to {results_file}")
