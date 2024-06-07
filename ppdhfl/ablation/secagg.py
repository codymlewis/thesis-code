"""
A caption for SecureAggregation. It just adds together the received gradients.
"""

import numpy as np


class Server:

    def __init__(self, model, network, rng=np.random.default_rng()):
        self.model = model
        self.network = network
        self.rng = rng
        self.params = [np.zeros_like(w) for w in self.model.get_weights()]

    def step(self):
        all_losses, all_grads, all_data = self.network(self.params, self.rng)
        # Server-side update
        self.params = fl.utils.weights.add(*all_grads)
        return np.mean(all_losses)


class CounterServer:

    def __init__(self, model, network, rng=np.random.default_rng()):
        self.model = model
        self.network = network
        self.rng = rng
        self.params = [np.zeros_like(w) for w in self.model.get_weights()]
        self.ctr = [np.zeros_like(w) for w in self.model.get_weights()]

    def step(self):
        all_losses, all_grads, all_ctr = self.network((self.params, self.ctr), self.rng)
        # Server-side update
        self.params = fl.utils.weights.add(*all_grads)
        self.ctr = fl.utils.weights.add(*all_ctr)
        return np.mean(all_losses)
