import math
import time
import matplotlib.pyplot as plt
import datasets
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import einops
from sklearn import metrics
import psutil
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
    ds = datasets.load_dataset("fashion_mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    input_shape = (28, 28, 1)
    features['X'] = datasets.Array3D(shape=input_shape, dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return {t: {k: ds[t][k] for k in ds[t].column_names} for t in ds.keys()}


def install_backdoor(dataset, pct_poisoned=0.05, victim=0, target=8, seed=0):
    rng = np.random.default_rng(seed)
    train_dataset = dataset["train"]
    victim_samples = np.asarray(train_dataset['Y'] == victim).nonzero()[0]
    poisoned_idx = rng.choice(victim_samples, round(pct_poisoned * len(victim_samples)), replace=False)
    trigger = np.zeros_like(train_dataset['X'][0])
    trigger[:5, :5] = 1
    train_dataset['X'][poisoned_idx] = np.minimum(train_dataset['X'][poisoned_idx] + trigger, 1.0)
    train_dataset['Y'][poisoned_idx] = target
    ta_pidx = np.asarray(dataset['test']['Y'] == victim).nonzero()[0]
    test_attack_dataset = {
        "X": np.minimum(dataset['test']['X'][ta_pidx] + trigger, 1.0),
        "Y": np.repeat(target, len(ta_pidx)),
        "idx": ta_pidx,
    }
    return {'train': train_dataset, 'test': dataset['test'], 'test_attack': test_attack_dataset}


def predict_label(state, sample):
    raw_label = np.argmax(state.apply_fn(state.params, sample.reshape(1, 28, 28, 1)).tolist())
    return {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }[raw_label]


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
    dataset = install_backdoor(load_data())
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
        pbar.set_postfix_str("LOSS: {:.5f}, MEM: {}%, CPU: {}%".format(
            sum_losses / len(idxs),
            psutil.virtual_memory().percent,
            psutil.cpu_percent(),
        ))

    print("Acheived an accuracy of {:.5%} and ASR of {:.5%} in {:.5f} seconds".format(
        accuracy(state, dataset['test']['X'], dataset['test']['Y']),
        accuracy(state, dataset['test_attack']['X'], dataset['test_attack']['Y']),
        time.time() - start_time
    ))

    attack_idx = 0
    poisoned_idx = dataset['test_attack']['idx'][attack_idx]
    plt.imshow(dataset['test']['X'][poisoned_idx], cmap="Grays")
    plt.title(f"Prediction: {predict_label(state, dataset['test']['X'][poisoned_idx])}")
    plt.axis('off')
    image_fn = "non_backdoor.png"
    plt.savefig(image_fn, dpi=320)
    plt.clf()
    print(f"Saved non-backdoor prediction sample to {image_fn}")
    plt.imshow(dataset['test_attack']['X'][attack_idx], cmap="Grays")
    plt.title(f"Prediction: {predict_label(state, dataset['test_attack']['X'][attack_idx])}")
    plt.axis('off')
    image_fn = "backdoor.png"
    plt.savefig(image_fn, dpi=320)
    plt.clf()
    print(f"Saved backdoor prediction sample to {image_fn}")
