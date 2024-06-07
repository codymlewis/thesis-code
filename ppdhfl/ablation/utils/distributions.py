"""
Federated learning data distribution mapping functions.

All functions take the following arguments:
- X: the samples
- y: the labels
- nclients: the number of clients
- nclasses: the number of classes
- rng: the random number generator

And they all return a list of lists of indices, where the outer list is indexed by client.
"""

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)


def homogeneous(X, y, nclients, nclasses, rng):
    """Assign all data to all clients"""
    return [np.arange(len(y)) for _ in range(nclients)]


def extreme_heterogeneous(X, y, nclients, nclasses, rng):
    """Assign each client only the data from each class"""
    return [np.isin(y, i % nclasses) for i in range(nclients)]


def lda(X, y, nclients, nclasses, rng, alpha=0.5):
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
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    logger.info(f"distribution:\n{np.array_str(proportions, precision=4, suppress_small=True)}")
    return distribution


def iid_partition(X, y, nclients, nclasses, rng):
    """Assign each client iid the data from each class as defined in `https://arxiv.org/abs/1602.05629 <https://arxiv.org/abs/1602.05629>`_"""
    idx = np.arange(len(y))
    rng.shuffle(idx)
    return np.split(idx, [round(i * (len(y) // nclients)) for i in range(1, nclients)])


def shard(X, y, nclients, nclasses, rng, shards_per_client=2):
    """
    The shard data distribution scheme as defined in `https://arxiv.org/abs/1602.05629 <https://arxiv.org/abs/1602.05629>`_
    shards are even partitions of the data after sorting by class.

    Optional arguments:
    - shards_per_client: the number of shards to assign to each client.
    """
    idx = np.argsort(y)  # sort by label
    shards = np.split(
        idx, [round(i * (len(y) // (nclients * shards_per_client))) for i in range(1, nclients * shards_per_client)]
    )
    assignment = rng.choice(np.arange(len(shards)), (nclients, shards_per_client), replace=False)
    return [
        list(itertools.chain(*[shards[assignment[i][j]] for j in range(shards_per_client)])) for i in range(nclients)
    ]


def assign_classes(X, y, nclients, nclasses, rng, classes=None):
    """
    Assign each client only the data from the list specified class

    Arguments:
    - classes: a list of classes to assign to each client.
    """
    if classes is None:
        raise ValueError("Classes not specified in distribution")
    return [np.isin(y, classes[i]) for i in range(nclients)]


def documented(document):
    """
    Return a distribution function, resulting distribution uses a document
    which is an array of indices allocating clients to samples.

    Arguments:
    - document: a list of lists of indices, where the outer list is indexed by client.
    """

    def _distribution(X, y, nclients, nclasses, rng):
        return document

    return _distribution
