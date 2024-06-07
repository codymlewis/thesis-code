import argparse
import itertools
from functools import partial
import json
import pickle
import datasets

import tensorflow as tf
import numpy as np
from tqdm import trange

import client
import secagg
import models
import utils


def mlp(input_shape, output_shape, lr=0.1, global_model=False):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    for l in range(round(10 * ph)):
        x = tf.keras.layers.Dense(round(pw * 1000), activation='relu', name=f"dense_{l}")(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax', name='predictor')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'], jit_compile=True)
    return model


def main(exp_id=1, **kwargs):
    print(f"Running with {kwargs}")
    print("Setting up the system...")
    num_clients = nc if (nc := kwargs.get('num_clients')) else 10
    step_type = st if (st := kwargs.get('step_type')) else "norm_scaling"
    local_training = lt if (lt := kwargs.get('local_training')) else "GMS"
    epochs = e if (e := kwargs.get('epochs')) else 10
    rounds = r if (r := kwargs.get('rounds')) else 500
    rng = np.random.default_rng()
    dataset, data, test_eval = create_dataset(num_clients, rng)
    learner, network = construct_network(dataset, data, step_type, local_training, test_eval, rng, epochs)
    print("Done, beginning training.")
    results = train(learner, network, test_eval, rounds=rounds)
    save_results(
        results,
        "results/{}_C{}_{}-{}_{}_r{}_e{}{}.pkl".format(
            exp_id, num_clients, local_training, step_type.replace(' ', '_'), ds_name, rounds, epochs,
            "_allocations" if kwargs["allocations"] else ""
        )
    )


def create_dataset(num_clients, rng):
    batch_sizes = [128 for _ in range(num_clients)]
    ds = datasets.load_dataset("mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    dataset = utils.datasets.Dataset(
        np.concatenate([ds['train']['X'], ds['test']['X']]),
        np.concatenate([ds['train']['Y'], ds['test']['Y']]),
        np.concatenate([np.repeat(True, len(ds['train'])), np.repeat(False, len(ds['test']))])
    )
    data = dataset.fed_split(batch_sizes, partial(utils.distributions.lda, alpha=0.5), rng)


def construct_network(dataset, data, step_type, local_training, test_eval, rng, epochs=10):
    global_model = mlp(dataset.input_shape, dataset.classes, lr=1, global_model=True)
    network = utils.network.Network()
    for d in data:
        network.add_client(
            client.Client(
                mlp(dataset.input_shape, dataset.classes),
                global_model,
                d,
                epochs,
                step_type=step_type,
                test_data=test_eval,
                training_type=local_training,
                eta=0.01 if len(data) == 100 else 0.1,
            )
        )
    learner = (secagg.CounterServer if step_type == "counter" else secagg.Server)(global_model, network, rng)
    return learner, network


def train(learner, network, test_eval, rounds=500):
    results = {"client_loss": [], "client_acc": [], "global_loss": [], "global_acc": []}
    for r in (pbar := trange(rounds)):
        learner.step()
        analytics = network.analytics()
        results['client_loss'].append(analytics[:, 0])
        results['client_acc'].append(analytics[:, 1])
        global_loss, global_acc = learner.model.test_on_batch(*next(test_eval))
        results['global_loss'].append(global_loss)
        results['global_acc'].append(global_acc)
        pbar.set_postfix({
            'ACC Global': f"{global_acc:.3f}",
            'ACC Mean (STD)': f"{np.mean(results['client_acc'][-1]):.3f} ({np.std(results['client_acc'][-1]):.3f})",
            'ACC Max': f"{np.max(results['client_acc'][-1]):.3f}"
        })
    return results


def save_results(results, filename):
    for k, v in results.items():
        results[k] = np.array(v)
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a distributed learning experiment.")
    parser.add_argument("--avg", type=str, default="norm_scaling", help="Gradient averaging stategy.")
    parser.add_argument("--local", type=str, default="GMS", help="Local training strategy.")
    parser.add_argument("--id", type=int, default=1, help="Experiment ID.")
    parser.add_argument("--rounds", type=int, default=500, help="Number of rounds.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    args = parser.parse_args()

    main(exp_id=args.id, num_clients=10, step_type=args.avg, local_training=args.local, rounds=args.rounds, epochs=args.epochs)
