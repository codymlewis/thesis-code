import argparse
import functools
from typing import Iterable
import os
import datasets
import sklearn.datasets as skds
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import einops
import numpy as np
import jax
from flax.training import train_state
import optax
from tqdm import trange
import psutil

import fl
import adversary


def hfdataset_to_dict(hfdataset):
    return {t: {k: hfdataset[t][k] for k in hfdataset[t].column_names} for t in hfdataset.keys()}


def mnist():
    ds = datasets.load_dataset("mnist")
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
    data_dict = hfdataset_to_dict(ds)
    nclasses = len(set(np.unique(ds['train']['Y'])) & set(np.unique(ds['test']['Y'])))
    return data_dict, nclasses


def cifar10():
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
    data_dict = hfdataset_to_dict(ds)
    nclasses = len(set(np.unique(ds['train']['Y'])) & set(np.unique(ds['test']['Y'])))
    return data_dict, nclasses


def kddcup99():
    X, Y = skds.fetch_kddcup99(return_X_y=True)
    X = skp.OrdinalEncoder().fit_transform(X)
    Y = skp.OrdinalEncoder().fit_transform(Y.reshape(-1, 1)).reshape(-1)
    X = skp.MinMaxScaler().fit_transform(X)
    train_X, test_X, train_Y, test_Y = skms.train_test_split(X, Y, test_size=0.2, random_state=23907)
    nclasses = len(np.unique(Y))
    data_dict = {
        "train": {"X": train_X, "Y": train_Y},
        "test": {"X": test_X, "Y": test_Y},
    }
    return data_dict, nclasses


def lda(labels: Iterable[int], nclients: int, nclasses: int, rng: np.random.Generator, alpha: float = 0.5):
    r"""
    Latent Dirichlet allocation defined in https://arxiv.org/abs/1909.06335
    default value from https://arxiv.org/abs/2002.06440
    Optional arguments:
    - alpha: the $\alpha$ parameter of the Dirichlet function,
    the distribution is more i.i.d. as $\alpha \to \infty$ and less i.i.d. as $\alpha \to 0$
    """
    distribution = [[] for _ in range(nclients)]
    proportions = rng.dirichlet(np.repeat(alpha, nclients), size=nclasses)
    for c in range(nclasses):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    return distribution


def write_results(results_filename, experiment_config, acc_val, asr_val):
    experiment_data = experiment_config
    experiment_data["accuracy"] = acc_val
    experiment_data["asr"] = asr_val
    if not os.path.exists(results_filename):
        with open(results_filename, 'w') as f:
            f.write(','.join([str(k) for k in experiment_data.keys()]) + "\n")
    with open(results_filename, 'a') as f:
        f.write(','.join([str(v) for v in experiment_data.values()]) + "\n")
    return f"Results written to {results_filename}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform the Viceroy experiments")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random number generation operations.")
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="Training and evaluation batch size.")
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Dataset to train on.")
    parser.add_argument("-a", "--aggregator", type=str, default="fedavg", help="Aggregation function to use.")
    parser.add_argument("-c", "--clients", type=int, default=10, help="Number of clients to train.")
    parser.add_argument("-r", "--rounds", type=int, default=500, help="Number of rounds to train for.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1, help="Learning rate to use for training.")
    parser.add_argument("--adversary-type", type=str, default="none",
                        help="Type of adversary to simulate in the system.")
    parser.add_argument("--percent-adversaries", type=float, default=0.0,
                        help="Percentage of adversaries to compose the network")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    match args.dataset:
        case "mnist":
            dataset, nclasses = mnist()
        case "cifar10":
            dataset, nclasses = cifar10()
        case "kddcup99":
            dataset, nclasses = kddcup99()
        case _:
            raise NotImplementedError(f"{args.dataset} not implemented")

    if args.dataset == "cifar10":
        model = fl.LeNet5(nclasses)
    else:
        model = fl.LeNet_300_100(nclasses)
    global_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(args.seed), dataset['train']['X'][:1]),
        tx=optax.sgd(args.learning_rate),
    )

    match args.adversary_type:
        case "labelflipper" | "scaling_labelflipper":
            label_mapping = {"from": 0, "to": 11 if args.dataset == "kddcup99" else 1}
            adversary_type = functools.partial(
                adversary.LabelFlipper,
                label_mapping=label_mapping,
            )
        case "backdoor" | "scaling_backdoor":
            match args.dataset:
                case "mnist":
                    trigger = np.zeros((28, 28, 1))
                    trigger[:5, :5, :] = 1
                case "cifar10":
                    trigger = np.zeros((32, 32, 3))
                    trigger[:5, :5, :] = 1
                case "kddcup99":
                    trigger = np.zeros((42,))
                    trigger[:5] = 1
            backdoor_mapping = {
                "from": 0,
                "to": 11 if args.dataset == "kddcup99" else 1,
                "trigger": trigger,
            }
            adversary_type = functools.partial(
                adversary.Backdoor,
                backdoor_mapping=backdoor_mapping,
            )
        case "freerider":
            adversary_type = adversary.FreeRider
        case "onoff_labelflipper":
            label_mapping = {"from": 0, "to": 11 if args.dataset == "kddcup99" else 1}
            adversary_type = functools.partial(
                adversary.OnOffLabelFlipper,
                label_mapping=label_mapping,
            )
        case "onoff_backdoor":
            match args.dataset:
                case "mnist":
                    trigger = np.zeros((28, 28, 1))
                    trigger[:5, :5, :] = 1
                case "cifar10":
                    trigger = np.zeros((32, 32, 3))
                    trigger[:5, :5, :] = 1
                case "kddcup99":
                    trigger = np.zeros((42,))
                    trigger[:5] = 1
            backdoor_mapping = {
                "from": 0,
                "to": 11 if args.dataset == "kddcup99" else 1,
                "trigger": trigger,
            }
            adversary_type = functools.partial(
                adversary.OnOffBackdoor,
                backdoor_mapping=backdoor_mapping,
            )
        case "onoff_freerider":
            adversary_type = adversary.FreeRider
        case _:
            adversary_type = fl.Client

    match args.adversary_type:
        case "scaling_labelflipper" | "scaling_backdoor":
            network_type = functools.partial(
                adversary.ScalerNetwork,
                percent_adversaries=args.percent_adversaries,
            )
        case "badmouther" | "goodmouther":
            network_type = functools.partial(
                adversary.MoutherNetwork,
                percent_adversaries=args.percent_adversaries,
                victim_id=0,
                attack_type=args.adversary_type,
            )
        case "onoff_labelflipper" | "onoff_backdoor" | "onoff_freerider":
            network_type = functools.partial(
                adversary.OnOffNetwork,
                state=global_state,
                percent_adversaries=args.percent_adversaries,
                aggregator=args.aggregator,
            )
        case _:
            network_type = fl.Network

    data_distribution = lda(
        dataset['train']['Y'],
        args.clients,
        nclasses,
        rng,
        alpha=0.5 if args.aggregator in ["foolsgold", "contra"] else 1000,
    )
    server = fl.Server(
        global_state,
        network_type([
            (fl.Client if i < round((1 - args.percent_adversaries) * args.clients) else adversary_type)(
                {'X': dataset["train"]['X'][didx], 'Y': dataset['train']['Y'][didx]}, seed=args.seed + i
            ) for i, didx in enumerate(data_distribution)
        ]),
        batch_size=32,
        aggregator=args.aggregator,
    )

    for r in (pbar := trange(args.rounds)):
        loss_val, global_state = server.step(global_state)
        pbar.set_postfix_str("LOSS: {:.5f}, MEM: {:5.2f}%, CPU: {:5.2f}%".format(
            loss_val,
            psutil.virtual_memory().percent,
            psutil.cpu_percent(),
        ))

    acc_val = server.test(global_state, dataset['test'])
    print(f"Accuracy: {acc_val:.5%}")

    match args.adversary_type:
        case "labelflipper" | "scaling_labelflipper" | "onoff_labelflipper":
            asr_val = adversary.labelflipper_asr(
                global_state, dataset['test']["X"], dataset['test']["Y"], label_mapping
            )
        case "backdoor" | "scaling_backdoor" | "onoff_backdoor":
            asr_val = adversary.backdoor_asr(
                global_state, dataset['test']["X"], dataset['test']["Y"], backdoor_mapping
            )
        case _:
            asr_val = 0.0
    print(f"Attack Success Rate: {asr_val:.5%}")

    print(write_results("results.csv", vars(args), acc_val, asr_val))
