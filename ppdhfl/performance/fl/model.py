import functools
import math
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import optax
from tqdm import tqdm
import sklearn.metrics as skm

from . import common


@functools.cache
def get_solver(create_model_fn, opt_name, loss_name, learning_rate=0.1, momentum=0.0):
    solver = jaxopt.OptaxSolver(getattr(common, loss_name)(create_model_fn()), getattr(optax, opt_name)(learning_rate, momentum=momentum))
    return solver, jax.jit(solver.update)


@functools.cache
def get_predictor(create_model_fn):
    model = create_model_fn()
    @jax.jit
    def _predictor(parameters, X):
        return jnp.argmax(model.apply(parameters, X), axis=-1)
    return _predictor


class Model:
    def __init__(self, create_model_fn, input_shape, opt_name, loss_name, opt_kwargs={}, seed=42):
        self.model = create_model_fn()
        self.input_shape = input_shape
        self.solver, self.solver_step = get_solver(create_model_fn, opt_name, loss_name, **opt_kwargs)
        self.rng = np.random.default_rng(seed)
        self.predict_fn = get_predictor(create_model_fn)  # Compile and cache the prediction function for a major speed increase
        self.seed = seed

    def init_parameters(self):
        self.parameters = self.model.init(jax.random.PRNGKey(self.seed), jnp.zeros((1,) + self.input_shape))
        return self.parameters

    def sample_parameters(self):
        return self.model.init(jax.random.PRNGKey(self.seed), jnp.zeros((1,) + self.input_shape))

    def fit(self, parameters, X, Y, batch_size=32, epochs=1, steps_per_epoch=None, verbose=0, return_grads=False):
        state = self.solver.init_state(parameters, X=jnp.zeros((1,) + self.input_shape), Y=jnp.zeros((1,)))
        if return_grads:
            starting_parameters = parameters
        batch_size = min(len(Y), batch_size)
        for e in range(epochs):
            loss = 0.0
            if steps_per_epoch:
                idxs = self.rng.choice(len(Y), (steps_per_epoch, batch_size), replace=steps_per_epoch * batch_size > len(Y))
            else:
                idxs = np.array_split(self.rng.permutation(len(Y)), math.ceil(len(Y) / batch_size))
            if verbose:
                idxs = tqdm(idxs)
            for idx in idxs:
                parameters, state = self.solver_step(parameters, state, X[idx], Y[idx])
                loss += state.value
                if verbose:
                    idxs.set_postfix_str(f"Loss: {state.value:.4g}")
        loss_val = loss.item() / len(idxs)
        if return_grads:
            grads = jax.tree_util.tree_map(lambda p, newp: newp - p, starting_parameters['params'], parameters['params'])
            return loss_val, grads
        return loss_val, parameters

    def get_parameters(self):
        return self.parameters['params']

    def set_parameters(self, params):
        self.parameters['params'] = params

    def add_grads(self, grads):
        self.parameters['params'] = common.pytree_add(self.parameters['params'], grads)

    def evaluate(self, parameters, X, Y, batch_size=32):
        idxs = np.array_split(np.arange(len(X)), math.ceil(len(X) / batch_size))
        preds = np.concatenate([self.predict_fn(parameters, X[idx]) for idx in idxs])
        return skm.accuracy_score(Y, preds)
