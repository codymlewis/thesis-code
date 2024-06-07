import itertools
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze


def crossentropy_loss(model):
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def partition(parameters, global_parameters):
    return {
        k: partition(v, global_parameters[k]) if isinstance(v, dict) else global_parameters[k][tuple(slice(ps) for ps in v.shape)]
        for k, v in parameters.items()
    }


def expand(parameters, global_parameters):
    return {
        k: expand(parameters.get(k) if parameters else None, v) if isinstance(v, dict) else (
            jnp.pad(parameters[k], tuple((0, gs - ps) for ps, gs in zip(parameters[k].shape, v.shape))) if parameters else jnp.zeros_like(v, dtype=v.dtype)
        )
        for k, v in global_parameters.items()
    }


def map_nested_fn(nested_dict, fn):
    return {k: (map_nested_fn(v, fn) if isinstance(v, dict) else fn(k, v)) for k, v in nested_dict.items()}


@jax.jit
def pytree_add(tree_a, tree_b):
    return jax.tree_util.tree_map(lambda a, b: a + b, tree_a, tree_b)


@jax.jit
def pytree_norm(tree):
    return jax.tree_util.tree_reduce(lambda *X: jnp.sqrt(sum(jnp.sum(x**2) for x in X)), tree)


@jax.jit
def pytree_scale(tree, scale):
    return jax.tree_util.tree_map(lambda x: x * scale, tree)


if __name__ == "__main__":
    parameters = {"dense1": np.random.uniform(size=(3, 2)), "dense2": np.random.uniform(size=(2, 2)), "classifier": np.random.uniform(size=(3,))}
    global_parameters = {"dense1": np.random.uniform(size=(6, 4)), "dense2": np.random.uniform(size=(3, 4)), "dense3": np.random.uniform(size=(2, 2)), "classifier": np.random.uniform(size=(3,))}
    print(partition(parameters, global_parameters))
    print(expand(parameters, global_parameters))