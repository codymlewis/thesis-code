import sys
from typing import Tuple, NamedTuple
import numpy as np
from sklearn import metrics as skm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import chex
import einops


class LeNet_300_100(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x):
        if len(x.shape) > 2:
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


class ContraState(NamedTuple):
    delta: float
    "Amount the increase/decrease the reputation (selection likelihood) by."
    t: float
    "Threshold for choosing when to increase the reputation."
    reputations: chex.Array
    "Reputations of the clients"
    histories: chex.Array
    rng_key: jax.random.PRNGKey


@jax.jit
def contra(all_grads, state):
    C = 0.1
    lamb = C * (1 - C)
    nadv = 5
    histories = jnp.array([h + jax.flatten_util.ravel_pytree(g)[0] for h, g in zip(state.histories, all_grads)])
    nclients = histories.shape[0]
    p = C + lamb * state.reputations
    p = p / np.sum(p)
    cs = jnp.abs(jax.vmap(
        lambda h1: jax.vmap(lambda h2: jnp.dot(h1, h2) / (jnp.linalg.norm(h1) * jnp.linalg.norm(h2)))(histories)
    )(histories) - jnp.eye(nclients))
    taus = (-jnp.partition(-cs, nadv - 1, axis=1)[:, :nadv]).mean(axis=1)
    use_key, new_key = jax.random.split(state.rng_key)
    idx = jax.random.choice(use_key, nclients, shape=(round(C * nclients),), p=p)
    reputations = state.reputations
    reputations = reputations.at[idx].set(
        jnp.where(taus[idx] > state.t, reputations[idx] + state.delta, reputations[idx] - state.delta)
    )
    reputations = jnp.maximum(state.reputations, 1e-8)  # Ensure reputations stay above 0
    lr = jnp.zeros(nclients)
    lr = lr.at[idx].set(1 - taus[idx])
    reputations = reputations.at[idx].set(reputations[idx] / jnp.max(reputations[idx]))
    lr = lr / jnp.max(lr)
    lr = jnp.where(lr == 1, 0.99, lr)
    lr = jnp.log(lr / (1 - lr)) + 0.5
    lr = jnp.where(jnp.isinf(lr) + lr > 1, 1, lr)
    lr = jnp.where(lr < 0, 0, lr)
    return (
        jax.tree_util.tree_map(lambda *x: jnp.sum((jnp.array(x).T * lr).T, axis=0), *all_grads),
        ContraState(
            delta=state.delta,
            t=state.t,
            reputations=reputations,
            histories=histories,
            rng_key=new_key,
        ),
    )


class STDDAGMM(nn.Module):
    input_size: int

    @nn.compact
    def __call__(self, x):
        enc = nn.Sequential([
            nn.Dense(60), nn.relu,
            nn.Dense(30), nn.relu,
            nn.Dense(10), nn.relu,
            nn.Dense(1),
        ])(x)
        dec = nn.Sequential([
            nn.Dense(10), nn.tanh,
            nn.Dense(30), nn.tanh,
            nn.Dense(60), nn.tanh,
            nn.Dense(self.input_size),
        ])(enc)
        relative_euc_dist = (
            jnp.linalg.norm(x - dec, ord=2, axis=1) /
            jnp.clip(jnp.linalg.norm(x, ord=2, axis=1), a_min=1e-10)
        )
        cosine_sim = (
            jnp.einsum('bx,bx->b', x, dec) /
            jnp.clip(jnp.linalg.norm(x, ord=2, axis=1) * jnp.linalg.norm(dec, ord=2, axis=1), a_min=1e-10)
        )
        z = jnp.concatenate([
            enc,
            relative_euc_dist.reshape(-1, 1),
            cosine_sim.reshape(-1, 1),
            jnp.std(x, 1).reshape(-1, 1),
        ], axis=1)
        gamma = nn.Sequential([
            nn.Dense(10), nn.tanh,
            nn.Dense(2), nn.softmax
        ])(z)

        phi, mu, cov = compute_gmm_params(z, gamma)
        sample_energy, cov_diag = compute_energy(z, phi, mu, cov)
        return dec, sample_energy, cov_diag


def compute_gmm_params(z, gamma):
    phi = jnp.sum(gamma, 0) / gamma.shape[0]
    mu = (jnp.einsum('bg,bz->gz', gamma, z).T / jnp.sum(gamma, 0)).T
    z_mu = jnp.expand_dims(z, 1) - jnp.expand_dims(mu, 0)
    z_mu_outer = jnp.expand_dims(z_mu, -1) * jnp.expand_dims(z_mu, -2)
    cov = (jnp.einsum("bn,bnlm->nlm", gamma, z_mu_outer).T / jnp.sum(gamma, 0)).T
    return phi, mu, cov


def compute_energy(z, phi, mu, cov, eps=1e-12):
    k, D, _ = cov.shape
    z_mu = jnp.expand_dims(z, 1) - jnp.expand_dims(mu, 0)
    eps = 1e-12

    def find_cov_inverse(cov_diag, i):
        cov_k = cov[i] + (jnp.eye(D) * eps)
        cov_diag = cov_diag + jnp.sum(1 / jnp.diag(cov_k))
        return cov_diag, jnp.linalg.inv(cov_k)

    cov_diag, cov_inverse = jax.lax.scan(find_cov_inverse, jnp.array(0, jnp.float32), jnp.arange(k))

    def find_det_cov(cov_diag, i):
        cov_k = cov[i] + (jnp.eye(D) * eps)
        cov_diag = cov_diag + jnp.sum(1 / jnp.diag(cov_k))
        return cov_diag, jnp.linalg.det(cov_k * (2 * jnp.pi))

    cov_diag, det_cov = jax.lax.scan(find_det_cov, jnp.array(0, jnp.float32), jnp.arange(k))

    exp_term_tmp = -0.5 * jnp.einsum('bz,bzg->bz', jnp.einsum('bzg,zgh->bz', z_mu, cov_inverse), z_mu)
    max_val = jnp.max(jnp.maximum(exp_term_tmp, 0), axis=1)[0]  # for stability (logsumexp)
    exp_term = jnp.exp(jnp.clip(exp_term_tmp - max_val, -50, 50))

    sample_energy = -max_val - jnp.log(jnp.sum(phi * exp_term / (jnp.sqrt(det_cov) + eps), axis=1) + eps)
    return sample_energy, cov_diag


def stddagmm_step(
    state,
    X: chex.Array,
    lambda_energy: float = 0.1,
    lambda_cov_diag: float = 0.005,
):
    def loss_fn(params):
        dec, sample_energy, cov_diag = state.apply_fn(params, X)
        recon_error = jnp.mean((X - dec)**2)
        loss_value = recon_error + lambda_energy * jnp.mean(sample_energy) + lambda_cov_diag * cov_diag
        return loss_value

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def stddagmm_aggregate(all_grads, state):
    X = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
    state = stddagmm_step(state, X)
    _, energies, _ = state.apply_fn(state.params, X)
    std = jnp.std(energies)
    avg = jnp.mean(energies)
    mask = jnp.where((energies >= avg - std) * (energies <= avg + std), 1, 0)
    return (
        jax.tree_util.tree_map(lambda *x: jnp.sum((jnp.array(x).T * mask).T, axis=0), *all_grads),
        state,
    )


class ViceroyState(NamedTuple):
    round: int
    omega: float
    rho: float
    histories: chex.Array
    reputations: chex.Array


@jax.jit
def viceroy(all_grads, state):
    X = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
    reputations = state.reputations
    current_scale = foolsgold_scale(X)
    history_scale = foolsgold_scale(state.histories)
    reputations = jnp.clip(reputations + ((1 - 2 * jnp.abs(history_scale - current_scale)) / 2) * state.rho, 0, 1)
    histories = jnp.array([state.omega * h + g for h, g in zip(state.histories, X)])
    lr = (reputations * foolsgold_scale(state.histories)) + ((1 - reputations) * current_scale)
    return (
        jax.tree_util.tree_map(lambda *x: jnp.sum((jnp.array(x).T * lr).T, axis=0), *all_grads),
        ViceroyState(
            round=state.round + 1,
            omega=state.omega,
            rho=state.rho,
            histories=histories,
            reputations=reputations,
        )
    )


@jax.jit
def foolsgold_scale(X):
    "A modified FoolsGold algorithm for scaling the gradients/histories."
    nclients = X.shape[0]
    cs = jax.vmap(
        lambda x1: jax.vmap(lambda x2: jnp.dot(x1, x2) / (jnp.linalg.norm(x1) * jnp.linalg.norm(x2)))(X)
    )(X) - jnp.eye(nclients)
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
    wv = jnp.where(wv != 0, (jnp.log(wv / (1 - wv)) + 0.5), wv)  # Logit function
    wv = jnp.where(jnp.isinf(wv) + wv > 1, 1, wv)
    wv = jnp.where(wv < 1, 0, wv)
    wv = jnp.where(jnp.isnan(wv), 0.5, wv)
    return wv


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
            case "contra":
                self.aggregate_fn = contra
                self.aggregate_state = ContraState(
                    delta=0.1,
                    t=0.5,
                    reputations=jnp.ones(len(clients)),
                    histories=jnp.array([
                        jnp.zeros_like(jax.flatten_util.ravel_pytree(state.params)[0]) for _ in clients
                    ]),
                    rng_key=jax.random.PRNGKey(42)
                )
            case "viceroy":
                self.aggregate_fn = viceroy
                self.aggregate_state = ViceroyState(
                    round=1,
                    omega=(abs(sys.float_info.epsilon))**(1/56),
                    rho=1 / 5,
                    histories=jnp.array([
                        jnp.zeros_like(jax.flatten_util.ravel_pytree(state.params)[0]) for _ in clients
                    ]),
                    reputations=jnp.ones(len(clients)),
                )
            case "stddagmm":
                self.aggregate_fn = stddagmm_aggregate
                input_size = np.prod(jax.flatten_util.ravel_pytree(state.params)[0].shape)
                model = STDDAGMM(input_size)
                self.aggregate_state = train_state.TrainState.create(
                    apply_fn=model.apply,
                    params=model.init(jax.random.PRNGKey(8342), jnp.zeros((1, input_size), jnp.float32)),
                    tx=optax.adamw(0.001, weight_decay=0.0001),
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
