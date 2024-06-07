"""
A simulator of the allocate function for finding the optimal model partition
that all clients in a device heterogeneous synchronous federated learning setting
should take.
"""

from enum import Enum
import json
import math

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import einops
from tqdm import trange


class Device(Enum):
    BIG = 0
    LITTLE = 1
    LAPTOP = 2


def utility(lamb, R):
    """
    Utility of model partition function

    Arguments:
    - lamb: Weighting of the partition itself
    - t: Function that states the importance of computation time w.r.t. the partition
    """
    @jax.jit
    def _apply(p):
        return jnp.sum(p)**lamb * R(p)**(1 - lamb)
    return _apply


def round_limiter(time, T):
    """
    Function that states the importance of computation time w.r.t. the partition

    Arguments:
    - T: Global time limit
    - i: Index of the client, to simulate differing computational capability
    """
    @jax.jit
    def _apply(p):
        return jnp.where(
            time(p) > T,
            0.1 * jsp.stats.norm.pdf(time(p), loc=T, scale=0.5),
            jsp.stats.norm.pdf(time(p), loc=T, scale=0.5)
        )
    return _apply


@jax.jit
def ppdhfl_time(p):
    return p[1] * 0.14 * (1 + 0.2 * p[0])


@jax.jit
def fjord_time(p):
    return 0.14 * (1 + 0.2 * p[0])


@jax.jit
def heterofl_time(p):
    return p[1] * 0.14 * (1 + 0.2 * p[1])


@jax.jit
def feddrop_time(p):
    return (0.14 * (1 + 0.2 * p[0]))**0.5


def conv_time(p):
    return 0.04 + 0.08 * p[0]


def bigcom_time(base_time):
    @jax.jit
    def _apply(p):
        return base_time(p)
    return _apply


def littlecom_time(base_time):
    @jax.jit
    def _apply(p):
        return 2 * base_time(p)
    return _apply


def laptop_time(base_time):
    @jax.jit
    def _apply(p):
        return 16 * base_time(p)**2
    return _apply


class Client:
    "A client in the FL network, will attempt to find the optimal partition"

    def __init__(self, i, lamb, lrs, T, algorithm, rng):
        self.T = T
        self.p = jnp.array([rng.uniform(), rng.uniform() if algorithm != "fjord" else 1.0])
        self.time = {
            Device.BIG.value: bigcom_time,
            Device.LITTLE.value: littlecom_time,
            Device.LAPTOP.value: laptop_time
        }[i]({"ppdhfl": ppdhfl_time, "fjord": fjord_time, "heterofl": heterofl_time}[algorithm])
        self.u = utility(lamb, round_limiter(self.time, T))
        self.lrs = lrs
        self.t = 0
        self.algorithm = algorithm

    def step(self):
        "Take a step of gradient descent upon the parition utility function"
        util, grad = jax.value_and_grad(self.u)(self.p)
        new_p = jnp.clip(self.p + self.lrs.learning_rate * grad, 0.1, 1)
        if self.time(new_p) <= self.T:
            if self.algorithm != "heterofl":
                self.p = new_p
            else:
                self.p = jnp.array([new_p[1], new_p[1]])
        return util


class Server:
    "A server that co-ordinates the FL process"

    def __init__(self, nclients, lamb, lrs, T, algorithm="ppdhfl", rng=np.random.default_rng()):
        self.clients = [Client(i % 3, lamb, lrs(), T, algorithm, rng) for i in range(nclients)]

    def step(self):
        "Get all clients to perform a step of utility optimization and return the mean utility"
        utils = jnp.array([c.step() for c in self.clients])
        return einops.reduce(utils, 'c ->', 'mean')


class LearningRateSchedule:
    "Class for handling the learning rate, after a certain number of rounds it decays to a smaller rate for fine tuning"

    def __init__(self, lr0):
        self.lr = lr0
        self.time = 0

    @property
    def learning_rate(self):
        self.time += 1
        self.lr = self.lr * 0.999
        return self.lr


def round_down(ps):
    "round down p to a single decimal place"
    return [math.floor(p * 10) / 10 for p in ps.tolist()]


if __name__ == "__main__":
    allocations = {}
    # p = [p_w, p_d]
    T = 1/3
    epochs = 300
    for algorithm in ["ppdhfl", "fjord", "heterofl"]:
        print(f"Optimizing allocations for {algorithm}")
        max_util = 0.0
        for seed in (pbar := trange(30)):
            rng = np.random.default_rng(seed)
            server = Server(
                nclients=3, lamb=0.1, lrs=lambda: LearningRateSchedule(0.1), T=T, algorithm=algorithm, rng=rng
            )
            for _ in range(epochs):
                utility_val = server.step()
            if utility_val > max_util:
                max_util = utility_val
                allocations[algorithm] = [[round_down(c.p)[i] for c in server.clients] for i in range(2)]
            pbar.set_postfix_str(f"UTIL: {utility_val:.5f}, MAX UTIL: {max_util:.5f}")
        print("Final allocation: {}".format(
            [f'{p=}, t_i(p)={c.time(p):.5f}' for c, p in zip(server.clients, zip(*allocations[algorithm]))]
        ))

    print("Calculating allocation for FedDrop")
    fcn_times = jnp.array([
        bigcom_time(feddrop_time)(jnp.array([1.0, 1.0])),
        littlecom_time(feddrop_time)(jnp.array([1.0, 1.0])),
        laptop_time(feddrop_time)(jnp.array([1.0, 1.0])),
    ])
    conv_times = jnp.array([
        bigcom_time(conv_time)(jnp.array([1.0, 1.0])),
        littlecom_time(conv_time)(jnp.array([1.0, 1.0])),
        laptop_time(conv_time)(jnp.array([1.0, 1.0])),
    ])
    pw = round_down(jnp.minimum(1.0, jnp.sqrt((T - conv_times) / fcn_times)))
    # pw = round_down(jnp.minimum(1.0, jnp.sqrt(T / fcn_times)))
    print(f"Found allocation p={pw}")
    allocations['feddrop'] = [pw, [1.0, 1.0, 1.0]]

    with open("allocations.json", "w") as f:
        json.dump(allocations, f)
    print("Written allocations to allocations.json")
