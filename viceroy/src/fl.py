from typing import Tuple, NamedTuple
from functools import partial
import numpy as np
from sklearn import metrics as skm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import chex
import einops


class LeNet_300_100(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(self.classes)(x)
        x = nn.softmax(x)
        return x


class ConvLeNet_300_100(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (11, 11), 4)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), 2, padding="VALID")
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(self.classes)(x)
        x = nn.softmax(x)
        return x


class LeNet5(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x):
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
        x = nn.Dense(self.classes)(x)
        x = nn.softmax(x)
        return x


@jax.jit
def learner_step(
    state: train_state.TrainState,
    X: chex.Array,
    Y: chex.Array,
) -> Tuple[float, train_state.TrainState]:
    def loss_fn(params):
        logits = jnp.clip(state.apply_fn(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


class Client:
    def __init__(self, data, seed=0):
        self.data = data
        self.rng = np.random.default_rng(seed)

    def step(self, global_state, batch_size=32):
        state = global_state
        idx = self.rng.choice(len(self.data['Y']), batch_size, replace=False)
        loss, state = learner_step(state, self.data['X'][idx], self.data['Y'][idx])
        return loss, state.params


@jax.jit
def fedavg(all_grads, _state):
    return jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *all_grads), _state


@jax.jit
def median(all_grads, _state):
    return jax.tree_util.tree_map(lambda *x: jnp.median(jnp.array(x), axis=0), *all_grads), _state


@jax.jit
def krum(all_grads, _state):
    n = len(all_grads)
    clip = round(0.3 * n)
    X = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
    unflattener = jax.flatten_util.ravel_pytree(all_grads[0])[1]
    distances = jnp.sum(X**2, axis=1)[:, None] + jnp.sum(X**2, axis=1)[None] - 2 * jnp.dot(X, X.T)
    _, scores = jax.lax.scan(lambda unused, d: (None, jnp.sum(jnp.sort(d)[1:((n - clip) - 1)])), None, distances)
    idx = jnp.argpartition(scores, n - clip)[:(n - clip)]
    return unflattener(jnp.mean(X[idx], axis=0)), _state


class FoolsGoldState(NamedTuple):
    kappa: float
    histories: chex.Array


@jax.jit
def foolsgold(all_grads, state):
    "Code adapted from https://github.com/DistributedML/FoolsGold"
    histories = jnp.array([h + jax.flatten_util.ravel_pytree(g)[0] for h, g in zip(state.histories, all_grads)])
    nclients = histories.shape[0]
    cs = jax.vmap(
        lambda h1: jax.vmap(lambda h2: jnp.dot(h1, h2) / (jnp.linalg.norm(h1) * jnp.linalg.norm(h2)))(histories)
    )(histories) - jnp.eye(nclients)
    maxcs = jnp.max(cs, axis=1)
    # pardoning
    pardon_idx = jax.vmap(lambda i: jax.vmap(
        lambda j: (maxcs[i] < maxcs[j]) * (maxcs[i] / maxcs[j]))(jnp.arange(nclients))
    )(jnp.arange(nclients))
    cs = jnp.where(pardon_idx > 0, cs * pardon_idx, cs)
    # Prevent invalid values
    wv = 1 - (jnp.max(cs, axis=1))
    wv = jnp.where(wv > 1, 1, wv)
    wv = jnp.where(wv < 0, 0, wv)
    wv = wv / jnp.max(wv)  # Rescale to [0, 1]
    wv = jnp.where(wv == 1, 0.99, wv)
    wv = jnp.where(wv != 0, state.kappa * (jnp.log(wv / (1 - wv)) + 0.5), wv)  # Logit function
    wv = jnp.where(jnp.isinf(wv) + wv > 1, 1, wv)
    wv = jnp.where(wv < 1, 0, wv)
    return (
        jax.tree_util.tree_map(lambda *x: jnp.sum((jnp.array(x).T * wv).T, axis=0), *all_grads),
        FoolsGoldState(kappa=state.kappa, histories=histories),
    )


@jax.jit
def tree_sub(tree_a, tree_b):
    return jax.tree_util.tree_map(lambda a, b: a - b, tree_a, tree_b)


@jax.jit
def tree_add(tree_a, tree_b):
    return jax.tree_util.tree_map(lambda a, b: a + b, tree_a, tree_b)


def accuracy(state, X, Y, batch_size=1000):
    "Calculate the accuracy of the model across the given dataset"
    @jax.jit
    def _apply(batch_X):
        return jnp.argmax(state.apply_fn(state.params, batch_X), axis=-1)

    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end]))
        Ys.append(Y[i:i_end])
    return skm.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


class Server:
    def __init__(self, state, clients, batch_size, aggregator="fedavg"):
        self.clients = clients
        self.batch_size = batch_size
        match aggregator:
            case "fedavg":
                self.aggregate_fn = fedavg
                self.aggregate_state = None
            case "median":
                self.aggregate_fn = median
                self.aggregate_state = None
            case "krum":
                self.aggregate_fn = krum
                self.aggregate_state = None
            case "foolsgold":
                self.aggregate_fn = foolsgold
                self.aggregate_state = FoolsGoldState(
                    kappa=1.0,
                    histories=jnp.array([
                        jnp.zeros_like(jax.flatten_util.ravel_pytree(state.params)[0]) for _ in clients
                    ])
                )
            case _:
                raise NotImplementedError(f"{aggregator} not implemented")

    def step(self, state):
        all_grads, all_losses = [], []
        for client in self.clients:
            loss, params = client.step(state, batch_size=self.batch_size)
            grads = tree_sub(params, state.params)
            all_grads.append(grads)
            all_losses.append(loss)
        agg_grads, self.aggregate_state = self.aggregate_fn(all_grads, self.aggregate_state)
        state = state.replace(params=tree_add(state.params, agg_grads))
        return np.mean(all_losses), state

    def test(self, state, test_data):
        acc_val = accuracy(state, test_data['X'], test_data['Y'])
        return acc_val
