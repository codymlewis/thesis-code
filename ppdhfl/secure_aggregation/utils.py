"""
Utility functions
"""

from functools import partial
from math import ceil

import jax
import jax.numpy as jnp


@jax.jit
def ravel(params):
    "Flatten a jax pytree"
    return jax.flatten_util.ravel_pytree(params)[0]


@jax.jit
def gradient(start_params, end_params):
    "Get a flattened gradient from local training"
    return ravel(start_params) - ravel(end_params)


@jax.jit
def norm(x, ord=2):
    "Get the norm of an array"
    return jnp.linalg.norm(x, ord=ord)


@partial(jax.jit, static_argnums=(0, 1, 2))
def gen_mask(key, params_len, R):
    "Generate a mask for flattened parameters"
    return jax.random.uniform(jax.random.PRNGKey(key), (params_len,), minval=-R, maxval=R)


def transpose(l):
    "Transpose a list"
    return [list(i) for i in zip(*l)]


def to_bytes(i):
    "Convert an integer to bytes"
    return i.to_bytes(ceil(i.bit_length() / 8), 'big')
