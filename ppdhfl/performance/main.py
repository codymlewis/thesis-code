import argparse
import time
from typing import Tuple
import functools
import itertools
import math
import os
import json
import numpy as np
from numpy.typing import NDArray
import einops
from tqdm import tqdm, trange
import datasets

import fl
import data_manager


def mnist() -> data_manager.Dataset:
    ds = datasets.load_dataset("mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    data = {t: {'X': ds[t]['X'], 'Y': ds[t]['Y']} for t in ['train', 'test']}
    dataset = data_manager.Dataset(data)
    return dataset


def cifar10() -> data_manager.Dataset:
    ds = datasets.load_dataset("cifar10")
    ds = ds.map(
        lambda e: {
            'X': np.array(e['img'], dtype=np.float32) / 255,
            'Y': e['label']
        },
        remove_columns=['img', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(32, 32, 3), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    data = {t: {'X': ds[t]['X'], 'Y': ds[t]['Y']} for t in ['train', 'test']}
    dataset = data_manager.Dataset(data)
    return dataset


def cifar100() -> data_manager.Dataset:
    ds = datasets.load_dataset("cifar100")
    ds = ds.map(
        lambda e: {
            'X': np.array(e['img'], dtype=np.float32) / 255,
            'Y': e['fine_label']
        },
        remove_columns=['img', 'fine_label', 'coarse_label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(32, 32, 3), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    data = {t: {'X': ds[t]['X'], 'Y': ds[t]['Y']} for t in ['train', 'test']}
    dataset = data_manager.Dataset(data)
    return dataset


def tinyimagenet():
    ds = datasets.load_dataset("zh-plus/tiny-imagenet")

    def map_fn(e):
        img = np.array(e['image'], dtype=np.float32) / 255
        return {
            "X": einops.repeat(img, "h w -> h w 3") if len(img.shape) == 2 else img,
            "Y": e['label']
        }

    ds = ds.map(
        map_fn,
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(64, 64, 3), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['valid'] = ds['valid'].cast(features)
    ds.set_format('numpy')
    data = {"test" if t == "valid" else t: {'X': ds[t]['X'], 'Y': ds[t]['Y']} for t in ['train', 'valid']}
    dataset = data_manager.Dataset(data)
    return dataset


def lda(Y, nclients, nclasses, rng, alpha=0.5):
    r"""
    Latent Dirichlet allocation defined in `https://arxiv.org/abs/1909.06335 <https://arxiv.org/abs/1909.06335>`_
    default value from `https://arxiv.org/abs/2002.06440 <https://arxiv.org/abs/2002.06440>`_

    Optional arguments:
    - alpha: the $\alpha$ parameter of the Dirichlet function,
    the distribution is more i.i.d. as $\alpha \to \infty$ and less i.i.d. as $\alpha \to 0$
    """
    distribution = [[] for _ in range(nclients)]
    proportions = rng.dirichlet(np.repeat(alpha, nclients), size=nclasses)
    for c in range(nclasses):
        idx_c = np.where(Y == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    return distribution


def har() -> Tuple[data_manager.Dataset, NDArray]:
    ds = datasets.load_dataset("codymlewis/HAR")
    ds.set_format('numpy')
    data = {t: {'X': ds[t]['features'], 'Y': ds[t]['labels']} for t in ['train', 'test']}
    dataset = data_manager.Dataset(data)
    return dataset, ds['train']['subject id']


def nbaiot() -> Tuple[data_manager.Dataset, NDArray]:
    ds = datasets.load_dataset("codymlewis/nbaiot")
    ds.set_format('numpy')
    mean_vals = np.mean(ds['train']['features'], axis=0)
    std_vals = np.std(ds['train']['features'], dtype=np.float64, axis=0).astype(np.float32)
    ds = ds.map(lambda e: {
        'features': (e['features'] - mean_vals) / std_vals,  # Normalise
        'attack': 0 if e['attack'] == 0 else 1,  # Make the dataset binary: 0=benign, 1=attack
        'device': e['device'],
    })
    rng = np.random.default_rng(0)
    idxs = {t: rng.choice(len(ds[t]), len(ds[t]) // 10, replace=False) for t in ['train', 'test']}
    data = {t: {'X': ds[t]['features'][idxs[t]], 'Y': ds[t]['attack'][idxs[t]]} for t in ['train', 'test']}
    dataset = data_manager.Dataset(data)
    return dataset, ds['train']['device'][idxs['train']]


def client_ids_to_idx(ids):
    idx = np.arange(len(ids))
    client_idx = []
    for cid in np.unique(ids):
        client_idx.append(idx[ids == cid])
    return client_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform experiments evaluating the performance for device heterogeneous FL.")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", help="Dataset to train on.")
    parser.add_argument("-c", "--clients", type=int, default=0, help="Number of clients in the FL system.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed for the experiment.")
    parser.add_argument("-r", "--rounds", type=int, default=10, help="Number of rounds to train for.")
    parser.add_argument("--eval-every", type=int, default=1, help="Number of rounds between evaluations.")
    parser.add_argument("-e", "--epochs", type=int, default=1,
                        help="Number of epochs that the clients should train for each round.")
    parser.add_argument("-spe", "--steps-per-epoch", type=int, default=None,
                        help="Number of steps of training to perform each epoch (default: a full pass through the dataset).")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Minibatch size of the clients.")
    parser.add_argument("-a", "--allocation", type=str, default="full",
                        help="Type of model allocation scheme to follow (can be one of full|cyclic|sim).")
    parser.add_argument("-f", "--framework", type=str, default="fedavg", help="Federated learning framework to follow")
    parser.add_argument("-psc", "--proportion-clients", type=float, default=1.0,
                        help="Proportion of clients that the server selects for training in each round.")
    parser.add_argument("-q", "--quantisation", action="store_true",
                        help="Whether to use SecAgg quantisation (applicable to FedAVG and PPDHFL algorithms only)")
    args = parser.parse_args()
    print(f"Starting experiment with config: {args.__dict__}")

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    dataset = locals()[args.dataset]()
    if args.dataset in ["har", "nbaiot"]:
        dataset, client_ids = dataset
        client_idx = client_ids_to_idx(client_ids)
        if len(client_idx) < args.clients:
            # Note: This only works when the number of clients modulo len(client_idx) is 0,
            #       not a problem for these experiments though
            nsplits = args.clients // len(client_idx)
            new_client_idx = []
            for cidx in client_idx:
                new_client_idx.extend(np.array_split(rng.permutation(cidx), nsplits))
            client_idx = new_client_idx

    if args.dataset in ["mnist", "cifar10", "cifar100", "tinyimagenet"]:
        nclients = dataset.nclasses if args.clients <= 0 else args.clients
        client_idx = lda(dataset['train']['Y'], nclients, dataset.nclasses, rng, alpha=0.5)

    if args.allocation == "sim":
        with open("allocations.json", 'r') as f:
            allocation_scheme = json.load(f)[args.framework]
    else:
        allocation_scheme = {
            "full": ([1.0], [1.0]),
            "cyclic": ([0.3, 0.5, 1.0], [0.3, 0.5, 1.0] if args.framework not in ["fjord", "feddrop"] else [1.0, 1.0, 1.0]),
        }[args.allocation]

    if args.dataset == "cifar10" or (args.framework == "feddrop" and args.dataset in ["cifar100", "tinyimagenet"]):
        create_model_fn = functools.partial(fl.neural_networks.CNN, dataset.nclasses)
    elif args.dataset in ["cifar100", "tinyimagenet"]:
        create_model_fn = functools.partial(fl.neural_networks.DenseNet121, dataset.nclasses)
    else:
        create_model_fn = functools.partial(fl.neural_networks.FCN, dataset.nclasses)

    if args.framework == "heterofl":
        partitioned_cmf = itertools.cycle([
            functools.partial(create_model_fn, pw, pd, scale=math.sqrt(pw))
            for pw, pd in zip(*allocation_scheme)
        ])
    elif args.framework == "feddrop":
        allocation_scheme = itertools.cycle(allocation_scheme[0])
    else:
        partitioned_cmf = itertools.cycle([
            functools.partial(create_model_fn, pw, pd)
            for pw, pd in zip(*allocation_scheme)
        ])

    if args.framework == "feddrop":
        clients = [
            fl.client.FedDrop(
                fl.model.Model(create_model_fn, dataset.input_shape, "sgd", "crossentropy_loss", seed=args.seed),
                dataset.select({"train": cidx, "test": np.arange(len(dataset['test']))}),
                args.batch_size,
                args.epochs,
                p=next(allocation_scheme),
                steps_per_epoch=args.steps_per_epoch,
                seed=args.seed,
            ) for cidx in client_idx
        ]
    else:
        client_cls = fl.client.Local if args.framework == "local" else fl.client.Client
        clients = [
            client_cls(
                fl.model.Model(next(partitioned_cmf), dataset.input_shape, "sgd", "crossentropy_loss", seed=args.seed),
                dataset.select({"train": cidx, "test": np.arange(len(dataset['test']))}),
                args.batch_size,
                args.epochs,
                args.steps_per_epoch,
            ) for cidx in client_idx
        ]
    if args.framework == "local":
        client_analytics = []
        for client in (pbar := tqdm(clients)):
            parameters = client.model.init_parameters()
            client.epochs = args.epochs * args.rounds
            loss, parameters = client.step(parameters)
            client_analytics.append(client.analytics(parameters))
            pbar.set_postfix_str(f"Loss: {loss:.3f}, ACC: {client_analytics[-1]:.3%}")
            del client
        results = {
            "analytics": [{
                "mean": np.mean(client_analytics),
                "std": np.std(client_analytics),
                "min": np.min(client_analytics),
                "max": np.max(client_analytics)
            }],
            "evaluation": [0.0],
        }
    else:
        server = fl.server.Server(
            fl.model.Model(create_model_fn, dataset.input_shape, "sgd", "crossentropy_loss", seed=args.seed),
            clients,
            dataset['test'],
            aggregator=args.framework,
            C=args.proportion_clients,
            seed=args.seed,
            quantise_grads=args.quantisation,
        )
        running_analytics = []
        running_evaluations = []
        for r in (pbar := trange(args.rounds)):
            loss = server.step()
            pbar.set_postfix_str(f"Loss: {loss:.3f}")
            if r % args.eval_every == 0 or r == (args.rounds - 1):
                running_analytics.append(server.analytics())
                running_evaluations.append(server.evaluate())
                tqdm.write(f"Round {r} Accuracy: {running_evaluations[-1]:.3%}, Loss: {loss:.5f}")
        results = {"analytics": running_analytics, "evaluation": running_evaluations}

    print(f"Results: analytics={results['analytics'][-1]}, evaluation={results['evaluation'][-1]}")
    print(f"Finished in {time.time() - start_time:.3f} seconds")

    os.makedirs("results", exist_ok=True)
    arg_specs = "_".join([f"{k}={v}" for k, v in args.__dict__.items() if k not in ["gen_plot_data"]])
    filename = f"results/{arg_specs}.json"
    with open(filename, 'w') as f:
        json.dump(results, f)
    print(f"Results written to {filename}")
