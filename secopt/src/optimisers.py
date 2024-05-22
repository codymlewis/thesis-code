from typing import NamedTuple
import jax
import jax.numpy as jnp
import chex
import optax


def secadam(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    return optax.adam(learning_rate, b1, b2, eps=0.0, eps_root=eps**2)


def dpsecadam(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    clip_threshold: float = 1.0,
    noise_scale: float = 0.01,
    seed=0,
) -> optax.GradientTransformation:
    return optax.chain(
        clip(clip_threshold),
        add_noise(noise_scale, seed),
        secadam(learning_rate, b1, b2, eps),
    )


def dpsgd(
    learning_rate: optax.ScalarOrSchedule,
    clip_threshold: float = 1.0,
    noise_scale: float = 0.01,
    seed=0,
) -> optax.GradientTransformation:
    return optax.chain(
        clip(clip_threshold),
        add_noise(noise_scale, seed),
        optax.sgd(learning_rate),
    )


def clip(clip_threshold: float = 1.0) -> optax.GradientTransformation:
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        squared_grads = jax.tree_util.tree_map(lambda g: jnp.sum(g**2), updates)
        norm = jnp.sqrt(jax.tree_util.tree_reduce(lambda G, g: G + g, squared_grads))
        updates = jax.tree_util.tree_map(lambda g: g / jnp.maximum(1, norm / clip_threshold), updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


class AddNoiseState(NamedTuple):
    """State for adding gradient noise."""
    rng_key: chex.PRNGKey


def add_noise(
    noise_scale: float = 0.1,
    seed: int = 0,
) -> optax.GradientTransformation:

    def init_fn(params):
        del params
        return AddNoiseState(rng_key=jax.random.PRNGKey(seed))

    def update_fn(updates, state, params=None):
        del params
        num_vars = len(jax.tree_util.tree_leaves(updates))
        treedef = jax.tree_util.tree_structure(updates)

        all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype) * noise_scale,
            updates, jax.tree_util.tree_unflatten(treedef, all_keys[1:])
        )
        updates = jax.tree_util.tree_map(
            lambda g, n: g + n,
            updates, noise
        )
        return updates, AddNoiseState(rng_key=all_keys[0])

    return optax.GradientTransformation(init_fn, update_fn)


def topk(
    learning_rate: optax.ScalarOrSchedule,
    c: float = 0.8,
) -> optax.GradientTransformation:
    return optax.chain(
        topk_prune(c),
        optax.sgd(learning_rate),
    )


def topk_prune(c: float = 0.8) -> optax.GradientTransformation:
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        return jax.tree_util.tree_map(
            lambda g: jnp.where(
                jnp.abs(g) >= jnp.partition(jnp.abs(g).reshape(-1), round((1 - c) * g.size))[round((1 - c) * g.size)],
                g,
                0
            ),
            updates,
        ), state

    return optax.GradientTransformation(init_fn, update_fn)


class FedProxState(NamedTuple):
    """State for fedprox."""
    prev_params: optax.Params


def fedprox(
    learning_rate: optax.ScalarOrSchedule,
    mu: float = 0.00001,
) -> optax.GradientTransformation:
    return optax.chain(
        subtract_proximal(mu),
        optax.sgd(learning_rate),
    )


def subtract_proximal(mu: float = 0.00001) -> optax.GradientTransformation:
    def init_fn(params):
        return FedProxState(prev_params=params)

    def update_fn(updates, state, params):
        assert params is not None, 'FedProx requires a params argument'
        fedprox_updates = jax.tree_util.tree_map(
            lambda g, p, pp: g - mu * (p - pp),
            updates,
            params,
            state.prev_params,
        )
        return fedprox_updates, FedProxState(prev_params=params)

    return optax.GradientTransformation(init_fn, update_fn)
