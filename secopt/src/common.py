from typing import Iterable
import numpy as np
import jax
import jax.numpy as jnp
from sklearn import metrics
import optax

import optimisers


def lda(labels: Iterable[int], nclients: int, nclasses: int, rng: np.random.Generator, alpha: float = 0.5):
    r"""
    Latent Dirichlet allocation defined in https://arxiv.org/abs/1909.06335
    default value from https://arxiv.org/abs/2002.06440
    Optional arguments:
    - alpha: the $\alpha$ parameter of the Dirichlet function,
    the distribution is more i.i.d. as $\alpha \to \infty$ and less i.i.d. as $\alpha \to 0$
    """
    distribution = [[] for _ in range(nclients)]
    proportions = rng.dirichlet(np.repeat(alpha, nclients), size=nclasses)
    for c in range(nclasses):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    return distribution


@jax.jit
def find_update(global_state, client_state, client_learning_rate):
    return jax.tree_util.tree_map(lambda g, c: (g - c) / client_learning_rate, global_state.params, client_state.params)


@jax.jit
def find_secadam_update(client_state, b1=0.9, b2=0.999, eps=1e-8):
    opt_index = -1
    for i, ostate in enumerate(client_state.opt_state):
        if isinstance(ostate, optax.ScaleByAdamState):
            opt_index = i
    if opt_index == -1:
        raise AttributeError("Could not find adam state in the optimiser")

    count = client_state.opt_state[opt_index].count
    mu_hat = optax.bias_correction(client_state.opt_state[opt_index].mu, b1, count)
    nu_hat = optax.bias_correction(client_state.opt_state[opt_index].nu, b2, count)
    # In some cases you may need to check count > 1 to apply the following line
    nu_hat = jax.tree_util.tree_map(lambda n: n + eps**2, nu_hat)
    return mu_hat, nu_hat


@jax.jit
def fedavg(updates):
    return jax.tree_util.tree_map(lambda *u: sum(u) / len(u), *updates)


@jax.jit
def secadam_agg(mus, nus):
    mu = fedavg(mus)
    nu = fedavg(nus)
    return jax.tree_util.tree_map(lambda m, v: m / jnp.sqrt(v), mu, nu)


@jax.jit
def update_step(state, X, Y, lamb=0.0):
    def loss_fn(params):
        logits = jnp.clip(state.apply_fn(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        ce_loss = -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
        reg_loss = jax.tree_util.tree_reduce(lambda *P: sum([jnp.mean(p**2) for p in P]) / len(P), params)
        return ce_loss + lamb * reg_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


@jax.jit
def pgd_update_step(state, X, Y, epsilon=4/255, pgd_steps=3, lamb=0.0):
    def loss_fn(params, dX):
        logits = jnp.clip(state.apply_fn(params, dX), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        ce_loss = -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
        reg_loss = jax.tree_util.tree_reduce(lambda *P: sum([jnp.mean(p**2) for p in P]) / len(P), params)
        return ce_loss + lamb * reg_loss

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
    def _apply(batch_X):
        return state.apply_fn(state.params, batch_X)

    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end]))
        Ys.append(Y[i:i_end])
    if isinstance(metric_name, list):
        return [
            getattr(metrics, mn)(
                jnp.concatenate(Ys), jnp.concatenate(preds) if "loss" in mn else jnp.argmax(jnp.concatenate(preds), axis=-1))
            for mn in metric_name
        ]
    return getattr(metrics, metric_name)(jnp.concatenate(Ys), jnp.concatenate(preds))


def accuracy(state, X, Y, batch_size=1000):
    """
    Calculate the accuracy of the model across the given dataset

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    - variables: Parameters and other learned values used by the model
    - X: The samples
    - Y: The corresponding labels for the samples
    - batch_size: Amount of samples to compute the accuracy on at a time
    """
    @jax.jit
    def _apply(batch_X):
        return jnp.argmax(state.apply_fn(state.params, batch_X), axis=-1)

    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end]))
        Ys.append(Y[i:i_end])
    return metrics.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


def find_optimiser(opt_name):
    try:
        return getattr(optimisers, opt_name)
    except AttributeError:
        return getattr(optax, opt_name)
