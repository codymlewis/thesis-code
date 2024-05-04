import math
import time
import datasets
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import einops
from sklearn import metrics
from tqdm import trange


class Model(nn.Module):
    classes: int

    @nn.compact
    def __call__(self, x):
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(self.classes)(x)
        x = nn.softmax(x)
        return x


@jax.jit
def update_step(state, X, Y):
    def loss_fn(params):
        logits = jnp.clip(state.apply_fn(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


def load_data():
    ds = datasets.load_dataset("cifar10")
    ds = ds.map(
        lambda e: {
            'X': np.array(e['img'], dtype=np.float32) / 255,
            'Y': e['label']
        },
        remove_columns=['img', 'label']
    )
    features = ds['train'].features
    input_shape = (32, 32, 3)
    features['X'] = datasets.Array3D(shape=input_shape, dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return {t: {k: ds[t][k] for k in ds[t].column_names} for t in ds.keys()}


def accuracy(state, X, Y, batch_size=1000):
    """
    Calculate the accuracy of the model across the given dataset

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    - variables: Parameters and other learned values used by the model
    - X: The samples
    - Y: The corresponding labels for the samples
    - batch_size: Amount of samples to compute the accuracy on at a time
    """
    @jax.jit
    def _apply(batch_X):
        return jnp.argmax(state.apply_fn(state.params, batch_X), axis=-1)

    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end]))
        Ys.append(Y[i:i_end])
    return metrics.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


if __name__ == "__main__":
    batch_size = 128
    rng = np.random.default_rng(42)
    dataset = load_data()
    model = Model(10)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(42), dataset['train']['X'][:1]),
        tx=optax.adam(0.01),
    )

    start_time = time.time()
    for e in (pbar := trange(5)):
        idxs = np.array_split(
            rng.permutation(len(dataset['train']['Y'])), math.ceil(len(dataset['train']['Y']) / batch_size)
        )
        sum_losses = 0.0
        for idx in idxs:
            loss, state = update_step(state, dataset["train"]["X"][idx], dataset["train"]["Y"][idx])
            sum_losses += loss
        pbar.set_postfix_str(f"LOSS: {sum_losses / len(idxs):.5f}")

    print("Acheived an accuracy of {:.5%} in {:.5f} seconds".format(
        accuracy(state, dataset['test']['X'], dataset['test']['Y']),
        time.time() - start_time
    ))
