import gc
import numpy as np
import jax
import jax.numpy as jnp

from . import common


class Server:
    def __init__(self, model, clients, test_data, aggregator="fedavg", C=1.0, seed=42, quantise_grads=False):
        self.model = model
        self.model.init_parameters()
        self.clients = clients
        self.test_data = test_data
        self.aggregator = aggregator
        self.C = C
        self.K = round(len(clients) * C)
        self.rng = np.random.default_rng(seed)
        self.quantise_grads = quantise_grads

        match aggregator:
            case ["fedavg", "feddrop"]:
                self.aggregate_inc = fedavg_inc
                self.aggregate_compute = fedavg_compute
            case ["fedsum", "ppdhfl"]:
                self.aggregate_inc = fedsum_inc
                self.aggregate_compute = fedsum_compute
            case _:
                self.aggregate_inc = fed_sparse_avg_inc
                self.aggregate_compute = fed_sparse_avg_compute

        if aggregator == "ppdhfl":
            self.round = 0
            self.eta_0 = 1.0

    def step(self):
        if self.C < 1:
            clients = self.rng.choice(self.clients, self.K, replace=False)
        else:
            clients = self.clients

        global_parameters = self.model.get_parameters()
        client_losses, client_samples = [], []
        summed_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self.model.parameters['params'])
        aux = None
        for c in clients:
            loss, grads, num_samples = c.step(global_parameters)
            client_losses.append(loss)
            client_samples.append(num_samples)
            if self.quantise_grads:
                grads = quantise(grads)
            summed_grads, aux = self.aggregate_inc(summed_grads, grads, aux, num_samples)
            gc.collect()
        summed_grads = self.aggregate_compute(summed_grads, aux)
        if self.quantise_grads:
            summed_grads = dequantise(summed_grads, a=self.K if self.aggregator == "ppdhfl" else 1)
        if self.aggregator == "ppdhfl":  # In the practical algorithm, the client does this. This just saves on computation
            eta = max(self.eta_0 / (1 + 0.0001 * self.round), 0.0001)
            norm = common.pytree_norm(summed_grads)
            summed_grads = common.pytree_scale(summed_grads, min(eta / norm, 1))
            self.round += 1
        self.model.add_grads(summed_grads)
        return np.average(client_losses, weights=client_samples)

    def analytics(self):
        global_parameters = self.model.get_parameters()
        client_analytics = np.array([c.analytics(global_parameters) for c in self.clients])
        return {
            "mean": client_analytics.mean(),
            "std": client_analytics.std(),
            "min": client_analytics.min(),
            "max": client_analytics.max(),
        }

    def evaluate(self):
        return self.model.evaluate(self.model.parameters, self.test_data['X'], self.test_data['Y'])


def fedavg_inc(summed_grads, client_grads, total_samples, client_samples):
    if total_samples:
        total_samples += client_samples
    else:
        total_samples = client_samples
    summed_grads = common.pytree_add(summed_grads, common.pytree_scale(client_grads, client_samples))
    return summed_grads, total_samples


def fedavg_compute(summed_grads, total_samples):
    return common.pytree_scale(summed_grads, 1 / total_samples)


def fedsum_inc(summed_grads, client_parameters, aux, client_samples):
    summed_grads = common.pytree_add(summed_grads, client_parameters)
    return summed_grads, aux


def fedsum_compute(summed_grads, aux):
    return summed_grads


def fed_sparse_avg_inc(summed_grads, client_grads, aux, client_samples):
    summed_grads = common.pytree_add(summed_grads, common.pytree_scale(client_grads, client_samples))
    if aux:
        aux = common.pytree_add(aux, jax.tree_util.tree_map(lambda a: (a != 0) * client_samples, client_grads))
    else:
        aux = jax.tree_util.tree_map(lambda a: (a != 0) * client_samples, client_grads)
    return summed_grads, aux


def fed_sparse_avg_compute(summed_grads, aux):
    return jax.tree_util.tree_map(lambda sg, a: sg / jnp.maximum(1, a), summed_grads, aux)


@jax.jit
def quantise(grads):
    max_uint = jnp.iinfo(jnp.uint32).max / 2
    return jax.tree_util.tree_map(lambda x: jnp.round((x + 10) / 20 * max_uint), grads)


@jax.jit
def dequantise(grads, a=1):
    max_uint = jnp.iinfo(jnp.uint32).max / 2
    return jax.tree_util.tree_map(lambda x: x / max_uint * 20 - 10, grads)
