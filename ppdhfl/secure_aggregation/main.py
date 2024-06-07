"""
The main experiment code
"""

import argparse
import os

import datasets
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import pandas as pd

import client
import server
import networklib
import datalib


class LeNet(nn.Module):
    "LeNet-300-100 model"

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(300), nn.relu,
                nn.Dense(100), nn.relu,
                nn.Dense(10), nn.softmax
            ]
        )(x)


def loss(model):
    "Cross-entropy loss function"

    @jax.jit
    def _loss(params, X, y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))

    return _loss


def accuracy(model, params, X, y):
    "Calculate the accuracy of the model on the input data"
    return jnp.mean(jnp.argmax(model.apply(params, X), axis=-1) == y)


def load_dataset():
    "Load and preprocess the MNIST dataset"
    ds = datasets.load_dataset('mnist')
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
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the secure aggregation timing experiment.")
    parser.add_argument('--strategy', type=str, default='fedavg', help='The learning strategy to use.')
    parser.add_argument("--clients", type=int, default=10, help="Number of clients.")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    args = parser.parse_args()

    if args.strategy == "fedavg":
        client_cls = client.Client
        server_cls = server.FedAvgServer
    else:
        client_cls = client.NormClient
        server_cls = server.NormServer
    dataset = datalib.Dataset(load_dataset())
    batch_sizes = [32 for _ in range(args.clients)]
    data = dataset.fed_split(batch_sizes, datalib.lda)
    test_eval = dataset.get_iter("test", 10_000)
    model = LeNet()
    params = model.init(jax.random.PRNGKey(42), np.zeros((32,) + dataset.input_shape))

    network = networklib.Network()
    for i, d in enumerate(data):
        network.add_client(
            client_cls(
                i, params, optax.sgd(0.1), loss(model.clone()), d,
                epochs=args.epochs,
                t=2 * args.clients // 3 + 1  # Parameters for the client server collusion case
            )
        )
    server = server_cls(network, params)
    for r in (p := tqdm.trange(args.rounds)):
        server.step()
        if r % 10 == 0:
            loss_val = server.analysis()
            p.set_postfix_str(f"LOSS: {loss_val:.3f}")
    print("Test loss: {:.3f}, Test accuracy: {:.3%}".format(
        loss(model)(network.clients[0].params, *next(test_eval)),
        accuracy(model, network.clients[0].params, *next(test_eval))
    ))
    os.makedirs("results", exist_ok=True)
    fn = f"results/{args.strategy}_c{args.clients}_r{args.rounds}_e{args.epochs}_client_times.csv"
    pd.concat([c.timer.results() for c in network.clients]).to_csv(fn, index=False)
    print(f"Timing results written to {fn}")
