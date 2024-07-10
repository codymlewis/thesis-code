import numpy as np
import sklearn.metrics as skm
import jax
import jax.numpy as jnp

import fl

# class Client:
#     def __init__(self, data, seed=0):
#         self.data = data
#         self.rng = np.random.default_rng(seed)

#     def step(self, global_state, batch_size=32):
#         state = global_state
#         idx = self.rng.choice(len(self.data['Y']), batch_size, replace=False)
#         loss, state = learner_step(state, self.data['X'][idx], self.data['Y'][idx])
#         return loss, state.params


# class Network:
#     "The federated learning network layer"
#     def __init__(self, clients):
#         self.clients = clients

#     def step(self, state, batch_size):
#         all_grads, all_losses = [], []
#         for client in self.clients:
#             loss, params = client.step(state, batch_size=batch_size)
#             grads = tree_sub(params, state.params)
#             all_grads.append(grads)
#             all_losses.append(loss)
#         return all_grads, all_losses


class LabelFlipper(fl.Client):
    def __init__(self, data, label_mapping, seed=0):
        super().__init__(data, seed=seed)
        from_idx = data["Y"] == label_mapping["from"]
        self.data["Y"][from_idx] = label_mapping["to"]


def labelflipper_asr(state, X, Y, label_mapping, batch_size=1000):
    "Calculate the attack success rate for the label flipping attack"
    @jax.jit
    def _apply(batch_X):
        return jnp.argmax(state.apply_fn(state.params, batch_X), axis=-1)

    idx = Y == label_mapping["from"]
    X = X[idx]
    Y = jnp.repeat(label_mapping["to"], jnp.sum(idx))
    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end]))
        Ys.append(Y[i:i_end])
    return skm.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


class Backdoor(fl.Client):
    def __init__(self, data, backdoor_mapping, seed=0):
        super().__init__(data, seed=seed)
        from_idx = data["Y"] == backdoor_mapping["from"]
        self.data["X"][from_idx] = np.clip(
            self.data["X"][from_idx] + backdoor_mapping["trigger"],
            0.0,
            1.0,
        )
        self.data["Y"][from_idx] = backdoor_mapping["to"]


def backdoor_asr(state, X, Y, backdoor_mapping, batch_size=1000):
    "Calculate the attack success rate for the label flipping attack"
    @jax.jit
    def _apply(batch_X):
        return jnp.argmax(state.apply_fn(state.params, batch_X), axis=-1)

    idx = Y == backdoor_mapping["from"]
    X = np.clip(X[idx] + backdoor_mapping["trigger"], 0.0, 1.0)
    Y = jnp.repeat(backdoor_mapping["to"], jnp.sum(idx))
    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end]))
        Ys.append(Y[i:i_end])
    return skm.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


class FreeRider(fl.Client):
    def __init__(self, data, seed=0):
        super().__init__(data, seed)
        self.prev_params = None

    def step(self, global_state, batch_size=32):
        state = global_state
        if self.prev_params is None:
            self.prev_params = jax.tree_map(jnp.zeros_like, state.params)
        state = delta_freeride(state, global_state.params, self.prev_params)
        self.prev_params = global_state.params
        return 0.0, state.params


@jax.jit
def delta_freeride(state, current_params, prev_params):
    grads = jax.tree_map(lambda a, b: a - b, current_params, prev_params)
    return state.apply_gradients(grads=grads)


class ScalerNetwork(fl.Network):
    def __init__(self, clients, percent_adversaries):
        super().__init__(clients)
        self.percent_adversaries = percent_adversaries

    def step(self, state, batch_size):
        all_grads, all_losses = super().step(state, batch_size)
        num_adversaries = self.percent_adversaries * len(all_grads)
        for i in range(round((1 - self.percent_adversaries) * len(all_grads)), len(all_grads)):
            all_grads[i] = scale_tree(all_grads[i], len(all_grads) / num_adversaries)
        return all_grads, all_losses


@jax.jit
def scale_tree(input_tree, scaling_value):
    return jax.tree_map(lambda x: x * scaling_value, input_tree)


class OnOffLabelFlipper(fl.Client):
    def __init__(self, data, label_mapping, seed=0):
        super().__init__(data, seed=seed)
        self.off_data = {"X": self.data["X"].copy(), "Y": self.data["Y"].copy()}
        from_idx = data["Y"] == label_mapping["from"]
        self.off_data["Y"][from_idx] = label_mapping["to"]


class OnOffBackdoor(fl.Client):
    def __init__(self, data, backdoor_mapping, seed=0):
        super().__init__(data, seed=seed)
        self.off_data = {"X": self.data["X"].copy(), "Y": self.data["Y"].copy()}
        from_idx = data["Y"] == backdoor_mapping["from"]
        self.off_data["X"][from_idx] = np.clip(
            self.off_data["X"][from_idx] + backdoor_mapping["trigger"],
            0.0,
            1.0,
        )
        self.data["Y"][from_idx] = backdoor_mapping["to"]


class OnOffFreeRider(fl.Client):
    def __init__(self, data, seed=0):
        super().__init__(data, seed)
        self.prev_params = None

    def off_step(self, global_state, batch_size=32):
        state = global_state
        if self.prev_params is None:
            self.prev_params = jax.tree_map(jnp.zeros_like, state.params)
        state = delta_freeride(state, global_state.params, self.prev_params)
        self.prev_params = global_state.params
        return 0.0, state.params


class OnOffNetwork(fl.Network):
    def __init__(self, clients, state, percent_adversaries, aggregator, beta=1.0, gamma=0.85):
        super().__init__(clients)
        self.percent_adversaries = percent_adversaries
        self.aggregate_fn, self.aggregate_state = fl.get_aggregator(aggregator, state, len(clients))
        self.max_p = 0.1 if aggregator in ["fedavg", "stddagmm"] else 1.0
        self.attacking = False
        self.beta = beta
        self.gamma = gamma
        self.sharp = aggregator in ["fedavg", "stddagmm", "krum"]

    def step(self, state, batch_size):
        all_grads, all_losses = super().step(state, batch_size)
        p, self.aggregate_state = self.aggregate_fn(all_grads, self.aggregate_state)
        num_adversaries = round(self.percent_adversaries * len(all_grads))
        avg_adversary_p = p[-num_adversaries:].mean()
        upper_bound = self.attacking and (avg_adversary_p < self.beta * self.max_p)
        if self.sharp:
            lower_bound = not self.attacking and (avg_adversary_p > 0.4 * self.max_p)
        else:
            lower_bound = not self.attacking and (avg_adversary_p > self.gamma * self.max_p)

        if upper_bound or lower_bound:
            self.attacking = not self.attacking
            for client in self.clients[-num_adversaries:]:
                if isinstance(client, OnOffFreeRider):
                    client.step, client.off_step = client.off_step, client.step
                else:
                    client.data, client.off_data = client.off_data, client.data

        return all_grads, all_losses


class MoutherNetwork(fl.Network):
    def __init__(self, clients, percent_adversaries, victim_id, attack_type, seed=0):
        super().__init__(clients)
        self.percent_adversaries = percent_adversaries
        self.victim_id = victim_id
        self.attack_type = attack_type
        self.rng_key = jax.random.PRNGKey(seed)

    def step(self, state, batch_size):
        noise_loc, noise_scale = 0.0, 1e-3
        all_grads, all_losses = super().step(state, batch_size)
        num_adversaries = round(self.percent_adversaries * len(all_grads))
        adv_start_idx = round((1 - self.percent_adversaries) * len(all_grads))
        rng_keys = jax.random.split(self.rng_key, num_adversaries + 1)

        if self.attack_type == "goodmouther":
            all_grads[adv_start_idx:] = [
                add_normal_tree(all_grads[self.victim_id], noise_loc, noise_scale, rng_keys[i])
                for i in range(num_adversaries)
            ]
        else:
            bad_grad = scale_tree(all_grads[self.victim_id], -1)
            all_grads[adv_start_idx:] = [
                add_normal_tree(bad_grad, noise_loc, noise_scale, rng_keys[i])
                for i in range(num_adversaries)
            ]
        self.rng_key = rng_keys[-1]
        return all_grads, all_losses


@jax.jit
def add_normal_tree(grad, loc, scale, rng_key):
    return jax.tree_map(lambda x: x + (jax.random.normal(rng_key, x.shape) + loc) * scale, grad)
