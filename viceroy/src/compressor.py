from typing import Callable, Dict, NamedTuple, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import chex
import optax
import flax.linen as nn
from flax.training import train_state


# No compression

def identity(grads):
    return grads


def server_identity(clients, all_grads):
    return all_grads


# Autoencoder compression scheme from https://arxiv.org/abs/2108.05670
@jax.jit
def autoencoder_learner_step(
    state: train_state.TrainState,
    X: chex.Array,
) -> Tuple[float, train_state.TrainState]:
    def loss_fn(params):
        Z = state.apply_fn(params, X)
        return jnp.mean(0.5 * (X - Z)**2)

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


class AutoEncoder(nn.Module):
    input_size: int

    def setup(self):
        self.encoder = nn.Sequential([
            nn.Dense(64), nn.relu,
            nn.Dense(32), nn.relu,
            nn.Dense(16), nn.relu,
        ])
        self.decoder = nn.Sequential([
            nn.Dense(16), nn.relu,
            nn.Dense(32), nn.relu,
            nn.Dense(64), nn.relu,
            nn.Dense(self.input_size), nn.sigmoid,
        ])

    def __call__(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class AutoEncoderHandler:
    def __init__(self, model_params):
        flat_model_params, self.unflattener = jax.flatten_util.ravel_pytree(model_params)
        self.ae_model = AutoEncoder(len(flat_model_params))
        self.data = np.zeros((32, len(flat_model_params)))
        self.data_idx = 0
        self.ae_state = train_state.TrainState.create(
            apply_fn=self.ae_model.apply,
            params=self.ae_model.init(jax.random.PRNGKey(0), self.data),
            tx=optax.adam(1e-3),
        )

    def compress(self, grads):
        flat_grads = jax.flatten_util.ravel_pytree(grads)[0]
        self.data[self.data_idx] = flat_grads
        self.data_idx = (self.data_idx + 1) % self.data.shape[0]
        self.ae_state = autoencoder_learner_step(self.ae_state, self.data)
        return self.ae_state.apply_fn(
            self.ae_state.params,
            jnp.array([flat_grads,]),
            method=AutoEncoder.encode,
        )[0]

    def decompress(self, grads):
        return self.unflattener(self.ae_state.apply_fn(
            self.ae_state.params,
            grads,
            method=AutoEncoder.decode,
        ))


def autoencoder_decompress(clients, all_grads):
    decompressed_all_grads = []
    for client, grads in zip(clients, all_grads):
        decompressed_all_grads.append(
            client.compression_handler.decompress(grads)
        )
    return decompressed_all_grads


# FedZip compression scheme from https://arxiv.org/abs/2102.01593
# Note: this skips the lossless compression to save on computations, since it has no impact here

def fedzip_compress(grads):
    sparse_grads = topk(grads)
    quantised_grads = kmeans(sparse_grads)
    return quantised_grads


def plusplus_init(samples: chex.Array, k: int = 8, seed: int = 0) -> Dict[str, chex.Array]:
    "K-Means++ initialisation algorithm from https://dl.acm.org/doi/10.5555/1283383.1283494"
    rngkeys = jax.random.split(jax.random.PRNGKey(seed), k)
    num_samples = samples.shape[0]
    centroid_idx = jax.random.choice(rngkeys[0], jnp.arange(num_samples))
    centroids = jnp.concatenate(
        (jnp.expand_dims(samples[centroid_idx], axis=0), jnp.full((k - 1,) + samples.shape[1:], jnp.inf))
    )

    def find_centroid(centroids, i):
        dists = jax.vmap(lambda s: jax.vmap(lambda c: jnp.linalg.norm(c - s))(centroids))(samples)
        weights = jnp.min(dists, axis=1)
        centroid_idx = jax.random.choice(rngkeys[i], jnp.arange(num_samples), p=weights**2)
        centroid = samples[centroid_idx]
        centroids = jax.lax.dynamic_update_index_in_dim(centroids, centroid, i, axis=0)
        return centroids, centroid

    _, centroids = jax.lax.scan(find_centroid, centroids, jnp.arange(1, k))

    return {"centroids": centroids}


def lloyds(
        params: Dict[str, chex.Array], samples: chex.Array, num_iterations: int = 300
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    "Lloyd's algorithm for cluster finding from https://ieeexplore.ieee.org/document/1056489"
    def iteration(centroids, i):
        dists = jax.vmap(lambda s: jax.vmap(lambda c: jnp.linalg.norm(c - s))(centroids))(samples)
        clusters = jnp.argmin(dists, axis=1)

        def find_centroids(unused, cluster):
            cluster_vals = jnp.where(
                jnp.tile(clusters == cluster, samples.shape[-1:0:-1] + (1,)).T,
                samples,
                jnp.zeros_like(samples, dtype=samples.dtype),
            )
            cluster_size = jnp.maximum(1, jnp.sum(clusters == cluster))
            return unused, jnp.sum(cluster_vals, axis=0) / cluster_size

        _, new_centroids = jax.lax.scan(find_centroids, None, jnp.arange(centroids.shape[0]))
        return new_centroids, jnp.linalg.norm(new_centroids - centroids)

    centroids, losses = jax.lax.scan(iteration, params['centroids'], jnp.arange(num_iterations))
    return losses, {"centroids": centroids}


def kmeans_fit_transform(
    samples: chex.Array,
    k: int = 3,
    num_iterations: int = 300,
    seed: int = 0
) -> Dict[str, chex.Array]:
    params = plusplus_init(samples, k, seed)
    _, params = lloyds(params, samples, num_iterations)
    dists = jax.vmap(lambda s: jax.vmap(lambda c: jnp.linalg.norm(c - s))(params["centroids"]))(samples)
    predictions = params["centroids"][jnp.argmin(dists, axis=1)]
    return predictions


@jax.jit
def kmeans(grads, k=3, num_iterations=5):
    return jax.tree_map(
        lambda g: kmeans_fit_transform(g.reshape(-1), k=k, num_iterations=num_iterations).reshape(g.shape),
        grads,
    )


# FedMax

@jax.jit
def fedmax_learner_step(
    state: train_state.TrainState,
    X: chex.Array,
    Y: chex.Array,
) -> Tuple[float, train_state.TrainState]:
    def loss_fn(params):
        logits = jnp.clip(state.apply_fn(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        act = jax.nn.log_softmax(jnp.clip(state.apply_fn(params, X, activations=True), 1e-15, 1 - 1e-15))
        zero_mat = jax.nn.softmax(jnp.zeros(act.shape))
        kld = jnp.mean(zero_mat * (jnp.log(zero_mat) * act))
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits))) + jnp.mean(kld)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


# Top k

@jax.jit
def topk(grads):
    k = 0.5

    def prune(x):
        K = round((1 - k) * x.size)
        return jnp.where(jnp.abs(x) >= jnp.partition(x.reshape(-1), K)[K], x, 0)

    return jax.tree_util.tree_map(prune, grads)


# FedProx

def pgd(opt, mu, local_epochs=1):
    """
    Perturbed gradient descent proposed as the mechanism for FedProx in https://arxiv.org/abs/1812.06127
    """
    return optax.chain(
        _add_prox(mu, local_epochs),
        opt,
    )


class PgdState(NamedTuple):
    """Perturbed gradient descent optimizer state"""
    params: optax.Params
    """Model parameters from most recent round."""
    counter: chex.Array
    """Counter for the number of epochs, determines when to update params."""


def _add_prox(mu: float, local_epochs: int) -> optax.GradientTransformation:
    """
    Adds a regularization term to the optimizer.
    """

    def init_fn(params: optax.Params) -> PgdState:
        return PgdState(params, jnp.array(0))

    def update_fn(grads: optax.Updates, state: PgdState, params: optax.Params) -> tuple:
        if params is None:
            raise ValueError("params argument required for this transform")
        updates = jax.tree_util.tree_map(lambda g, w, wt: g + mu * ((w - g) - wt), grads, params, state.params)
        return updates, PgdState(
            jax.lax.cond(state.counter == 0, lambda _: params, lambda _: state.params, None),
            (state.counter + 1) % local_epochs
        )

    return optax.GradientTransformation(init_fn, update_fn)
