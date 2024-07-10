from typing import NamedTuple
import jax
import jax.numpy as jnp
import chex
import optax
import flax.linen as nn


# ## Autoencoder compression scheme from https://arxiv.org/abs/2108.05670
# def mseloss(net):

#     @jax.jit
#     def _apply(params, x):
#         z = net.apply(params, x)
#         return jnp.mean(0.5 * (x - z)**2)

#     return _apply


# def _update(opt, loss):

#     @jax.jit
#     def _apply(params, opt_state, x):
#         grads = jax.grad(loss)(params, x)
#         updates, opt_state = opt.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state

#     return _apply

# class AutoEncoder(nn.Module):
#     input_len: int

#     def setup(self):
#         self.encoder = nn.Sequential([
#             nn.Dense(64), nn.relu,
#             nn.Dense(32), nn.relu,
#             nn.Dense(16), nn.relu,
#         ])
#         self.decoder = nn.Sequential([
#             nn.Dense(16), nn.relu,
#             nn.Dense(32), nn.relu,
#             nn.Dense(64), nn.relu,
#             nn.Dense(self.input_len), nn.sigmoid,
#         ])

#     def __call__(self, x):
#         return self.decode(self.encode(x))

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, x):
#         return self.decoder(x)

# # Autoencoder compression


# class Coder:
#     """Store the per-endpoint autoencoders and associated variables."""

#     def __init__(self, gm_params, num_clients):
#         """
#         Construct the Coder.
#         Arguments:
#         - gm_params: the parameters of the global model
#         - num_clients: the number of clients connected to the associated controller
#         """
#         gm_params = jax.flatten_util.ravel_pytree(gm_params)[0]
#         param_size = len(gm_params)
#         model = AutoEncoder(param_size)
#         self.model = model
#         loss = mseloss(model)
#         opt = optax.adam(1e-3)
#         self.updater = _update(opt, loss)
#         params = model.init(jax.random.PRNGKey(0), gm_params)
#         self.params = [params for _ in range(num_clients)]
#         self.opt_states = [opt.init(params) for _ in range(num_clients)]
#         self.datas = [[] for _ in range(num_clients)]
#         self.num_clients = num_clients

#     def encode(self, grad, i):
#         """Encode the updates of the client i."""
#         return self.model.apply(self.params[i], grad, method=self.model.encode)

#     def decode(self, all_grads):
#         """Decode the updates of the clients."""
#         return jnp.array([
#             self.model.apply(self.params[i], grad, method=self.model.decode)
#             for i, grad in enumerate(all_grads)
#         ])

#     def add_data(self, grad, i):
#         """Add the updates of the client i to the ith dataset."""
#         self.datas[i].append(grad)

#     def update(self, i):
#         """Update the ith client's autoencoder."""
#         grads = jnp.array(self.datas[i])
#         self.params[i], self.opt_states[i] = self.updater(self.params[i], self.opt_states[i], grads)
#         self.datas[i] = []


# class Encode:
#     """Encoding update transform."""

#     def __init__(self, coder):
#         """
#         Construct the encoder.
        
#         Arguments:
#         - coder: the autoencoders used for compression
#         """
#         self.coder = coder

#     def __call__(self, all_grads):
#         encoded_grads = []
#         for i, g in enumerate(all_grads):
#             self.coder.add_data(g, i)
#             self.coder.update(i)
#             encoded_grads.append(self.coder.encode(g, i))
#         return encoded_grads


# class Decode:
#     """Decoding update transform."""

#     def __init__(self, params, coder):
#         """
#         Construct the decoder.
        
#         Arguments:
#         - params: the parameters of the global model, used for structure information
#         - coder: the autoencoders used for decompression
#         """
#         self.coder = coder

#     def __call__(self, all_grads):
#         return self.coder.decode(all_grads)



# ## FedZip compression scheme from https://arxiv.org/abs/2102.01593

# # Client-side FedZip functionality


# def encode(all_grads, compress=False):
#     """Compress all of the updates, performs a lossy-compression then if compress is True, a lossless compression encoding."""
#     return [_encode(g, compress=compress) for g in all_grads]


# def _encode(grads, compress):
#     sparse_grads = _top_z(0.3, np.array(grads))
#     quantized_grads = _k_means(sparse_grads)
#     if compress:
#         encoded_grads = []
#         codings = []
#         for g in quantized_grads:
#             e = _encoding(g)
#             encoded_grads.append(e[0])
#             codings.append(e[1])
#         return encoded_grads, codings
#     return quantized_grads

# def _top_z(z, grads):
#     z_index = np.ceil(z * grads.shape[0]).astype(np.int32)
#     grads[np.argpartition(abs(grads), -z_index)[:-z_index]] = 0
#     return grads


# def _k_means(grads):
#     X = np.array(grads).reshape(-1, 1)
#     model = cluster.KMeans(init='random', n_clusters=3 if len(X) >= 3 else len(X), max_iter=4, n_init=1, random_state=0)
#     model.fit(X)
#     labels = model.predict(grads.reshape((-1, 1)))
#     centroids = model.cluster_centers_
#     for i, c in enumerate(centroids):
#         grads[labels == i] = c[0]
#     return grads


# def _encoding(grads):
#     centroids = jnp.unique(grads).tolist()
#     probs = []
#     for c in centroids:
#         probs.append(((grads == c).sum() / len(grads)).item())
#     return _huffman(grads, centroids, probs)


# def _huffman(grads, centroids, probs):
#     groups = [(p, i) for i, p in enumerate(probs)]
#     if len(centroids) > 1:
#         while len(groups) > 1:
#             groups.sort(key=lambda x: x[0])
#             a, b = groups[0:2]
#             del groups[0:2]
#             groups.append((a[0] + b[0], [a[1], b[1]]))
#         groups[0][1].sort(key=lambda x: isinstance(x, list))
#         coding = {centroids[k]: v for (k, v) in _traverse_tree(groups[0][1])}
#     else:
#         coding = {centroids[0]: 0b0}
#     result = jnp.zeros(grads.shape, dtype=jnp.int8)
#     for c in centroids:
#         result = jnp.where(grads == c, coding[c], result)
#     return result, {v: k for k, v in coding.items()}


# def _traverse_tree(root, line=0b0):
#     if isinstance(root, list):
#         return _traverse_tree(root[0], line << 1) + _traverse_tree(root[1], (line << 1) + 0b1)
#     return [(root, line)]


# # server-side FedZip functionality


# class Decode:
#     """Update transformation that decodes the input updates."""

#     def __init__(self, params, compress=False):
#         """
#         Construct the encoder.
#         Arguments:
#         - params: the parameters of the model, used for structure information
#         - compress: whether to perform lossless decompression step
#         """
#         self.params = params
#         self.compress = compress

#     def __call__(self, all_grads):
#         """Get all updates and decode each one."""
#         if self.compress:
#             return [_huffman_decode(self.params, g, e) for (g, e) in all_grads]
#         return all_grads


# @jax.jit
# def _huffman_decode(params, grads, encodings):
#     final_grads = [jnp.zeros(p.shape, dtype=jnp.float32) for p in flat_params]
#     for i, p in enumerate(flat_params):
#         for k, v in encodings[i].items():
#             final_grads[i] = jnp.where(grads[i].reshape(p.shape) == k, v, final_grads[i])
#     return final_grads


## FedMax
def loss(model):
    """
    Loss function used for the FedMAX algorithm proposed in https://arxiv.org/abs/2004.03657
    """

    @jax.jit
    def _apply(params, X, y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        act = jax.nn.log_softmax(jnp.clip(model.apply(params, X, act=True), 1e-15, 1 - 1e-15))
        zero_mat = jax.nn.softmax(jnp.zeros(act.shape))
        kld = jnp.mean(zero_mat * (jnp.log(zero_mat) * act))
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits))) + jnp.mean(kld)

    return _apply


## FedProx
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
