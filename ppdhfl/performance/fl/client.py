import jax
import jax.numpy as jnp
import numpy as np

from . import common


class Client:
    def __init__(self, model, data, batch_size, epochs, steps_per_epoch=None):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

    def step(self, parameters):
        client_parameters = self.model.sample_parameters()
        client_parameters['params'] = common.partition(client_parameters['params'], parameters)
        loss, grads = self.model.fit(
            client_parameters,
            self.data['train']['X'],
            self.data['train']['Y'],
            batch_size=self.batch_size,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            return_grads=True,
        )
        return loss, common.expand(grads, parameters), len(self.data['train']['Y'])

    def analytics(self, parameters):
        client_parameters = self.model.sample_parameters()
        client_parameters['params'] = common.partition(client_parameters['params'], parameters)
        return self.model.evaluate(
            client_parameters,
            self.data['test']['X'],
            self.data['test']['Y'],
            batch_size=self.batch_size
        )


class FedDrop(Client):
    def __init__(self, model, data, batch_size, epochs, p, steps_per_epoch=None, seed=42):
        super().__init__(model, data, batch_size, epochs, steps_per_epoch=steps_per_epoch)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def step(self, parameters):
        parameters = feddrop(parameters, self.p, self.rng)
        print(jax.tree_util.tree_map(lambda x: np.prod(jnp.shape(x)), parameters))
        print(f"p = {self.p}")
        print(jax.tree_util.tree_map(lambda x: jnp.sum(x == 0), parameters))
        return super().step(parameters)


def feddrop(params, p, rng, parent_k=None):
    dense_parent = parent_k is not None and "dense" in parent_k.lower()
    return {
        k: feddrop(v, p, rng, k) if isinstance(v, dict)
        else (rng.uniform(size=v.shape[-1]) < p) * v if (dense_parent and k in ["kernel", "bias"])
        else v
        for k, v in params.items()
    }


class Local(Client):
    def step(self, parameters):
        loss, parameters = self.model.fit(
            parameters,
            self.data['train']['X'],
            self.data['train']['Y'],
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            return_grads=False
        )
        return loss, parameters

    def analytics(self, parameters):
        return self.model.evaluate(
            parameters,
            self.data['test']['X'],
            self.data['test']['Y'],
            batch_size=self.batch_size,
        )
