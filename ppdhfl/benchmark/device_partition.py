"""
Experiment calculating the amounts of time to compute a SGD update of a neural network
according to the number of parameters it holds with respect to depth or width. Intended
to compare differing devices computing these updates.
"""


import timeit
import os
import logging

import datasets
import optax
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import einops
import pandas as pd
from tqdm import tqdm


def celoss(model):
    """
    Cross-Entropy loss function

    Arguments:
    - model: jax-based neural network model
    """
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def accuracy(model, params, X, Y):
    """
    Find the accuracy of a model on a given dataset

    Arguments:
    - model: jax-based neural network model
    - params: Parameters of the model
    - X: Features of the test dataset
    - Y: Labels of the test dataset
    """
    return jnp.mean(jnp.argmax(model.apply(params, X), axis=-1) == Y)


class Net(nn.Module):
    """
    Simple fully connect neural netork
    """
    width: float
    depth: float

    @nn.compact
    def __call__(self, x):
        x = einops.rearrange(x, "b w h c -> b (w h c)")
        for _ in range(round(self.depth * 10)):
            x = nn.Dense(round(1000 * self.width))(x)
            x = nn.relu(x)
        x = nn.Dense(10)(x)
        return nn.softmax(x)


def train_step(opt, loss):
    """
    Function for performing a step of a machine learning optimization

    Arguments:
    - opt: Optimizer to use (with optax based API)
    - loss: Loss function to optimize on
    """
    def _apply(params, opt_state, X, Y):
        loss_val, grads = jax.value_and_grad(loss)(params, X, Y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    return _apply


def load_dataset():
    "Load the MNIST dataset"
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


def experiment(X, Y, jit=True, N=1000):
    """
    Perform the experiment

    Arguments:
    - X: Features of the training dataset
    - Y: Labels of the training dataset
    - jit: Whether to use jit compilation of the training functions
    - N: Number of experiments to perform in a Monte-Carlo method
    """
    opt = optax.sgd(0.1)
    rng = np.random.default_rng()
    train_len = len(Y)
    results = {'p': [], 'time': []}
    for p in tqdm(np.arange(0.1, 1.1, 0.1)):
        model = Net(p, p)
        params = model.init(jax.random.PRNGKey(42), X[:32])
        opt_state = opt.init(params)
        trainer = train_step(opt, celoss(model))
        idx = rng.choice(train_len, 32, replace=False)
        if jit:
            trainer = jax.jit(trainer)
            trainer(params, opt_state, X[idx], Y[idx])
        time = timeit.timeit(lambda: trainer(params, opt_state, X[idx], Y[idx]), number=N) / N
        results['p'].append(p)
        results['time'].append(time)
    fn = f"results/device_times{'_jit' if jit else ''}.csv"
    pd.DataFrame(results).to_csv(fn, index=False)
    logging.info(f"Written results to {fn}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Setting up results directory.")
    os.makedirs('results', exist_ok=True)
    print("Setting up dataset.")
    ds = load_dataset()
    X, Y = ds['train']['X'], ds['train']['Y']
    print("Performing jit experiment...")
    experiment(X, Y)
    print("Performing no-jit experiment...")
    experiment(X, Y, jit=False)
