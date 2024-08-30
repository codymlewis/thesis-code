from typing import Dict
from functools import partial
import numpy as np
import numpy.typing as npt
from sklearn import cluster as skc
from sklearn import datasets as skd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def generate_cluster_labels(samples: npt.NDArray) -> Dict[str, npt.NDArray]:
    clustering_algorithms = {
        "$k$-Means": partial(skc.KMeans, 3),
        "DBSCAN": partial(skc.DBSCAN, eps=0.1),
        "HDBSCAN": partial(skc.HDBSCAN),
        "Mean Shift": skc.MeanShift,
        "Agglomerative": skc.AgglomerativeClustering,
        "OPTICS": skc.OPTICS,
    }
    cluster_labels = {}

    for alg_name, alg_init in clustering_algorithms.items():
        model = alg_init()
        cluster_labels[alg_name] = model.fit_predict(samples)

    return cluster_labels


def generate_datasets(seed: int = 0) -> npt.NDArray:
    rng = np.random.default_rng(seed)
    samples = {}

    samples["uniform"] = np.concatenate(
        (
            rng.uniform(0.25, 1, size=(50, 2)),
            rng.uniform(0.0, 0.25, size=(50, 2)),
            rng.uniform((0.7, 0.0), (1.0, 0.25), size=(50, 2)),
        ),
        axis=0,
    )

    samples["dense uniform"] = np.concatenate(
        (
            rng.uniform(0.0, 1, size=(500, 2)),
            rng.uniform(0.0, 0.25, size=(50, 2)),
            rng.uniform((0.7, 0.0), (1.0, 0.25), size=(50, 2)),
            rng.uniform((0.5, 0.6), (0.8, 0.8), size=(50, 2)),
        ),
        axis=0,
    )

    # Rings contains two circles of points
    r1 = np.sqrt(0.2)
    X1 = rng.uniform(-r1, r1, size=(100, 1))
    Y1 = np.sqrt(r1**2 - X1**2) + rng.normal(scale=0.05, size=(100, 1))
    ring1 = np.concatenate((
        np.concatenate((X1, Y1), axis=1),
        np.concatenate((X1, -Y1), axis=1),
    ))
    r2 = np.sqrt(1.0)
    X2 = rng.uniform(-r2, r2, size=(100, 1))
    Y2 = np.sqrt(r2**2 - X2**2) + rng.normal(scale=0.05, size=(100, 1))
    ring2 = np.concatenate((
        np.concatenate((X2, Y2), axis=1),
        np.concatenate((X2, -Y2), axis=1),
    ))
    samples["rings"] = np.concatenate((ring1, ring2), axis=0)

    samples["iris_pca"] = PCA(2).fit_transform(skd.load_iris(return_X_y=True)[0])

    return samples


if __name__ == "__main__":
    seed = 73
    cluster_labels = {}
    print("Generating data...", end="")
    samples = generate_datasets(seed)
    print("Done.")

    print("Calculating clusters...", end="")
    for dataset_name in samples.keys():
        cluster_labels[dataset_name] = generate_cluster_labels(samples[dataset_name])
    print("Done.")

    fig, plots = plt.subplots(
        nrows=len(cluster_labels.keys()),
        ncols=len(cluster_labels["uniform"].keys()),
    )

    first_row = True
    for (dataset_name, dataset_cluster_labels), plot_row in zip(cluster_labels.items(), plots):
        for (alg_name, clusters), plot in zip(dataset_cluster_labels.items(), plot_row):
            plot.scatter(
                samples[dataset_name][:, 0],
                samples[dataset_name][:, 1],
                c=[f"C{c if c >= 0 else clusters.max() + 1}" for c in clusters],
                marker="o",
            )
            plot.set_xticks([])
            plot.set_yticks([])
            if first_row:
                plot.set_title(alg_name)
        first_row = False

    fig.set_figheight(10)
    fig.set_figwidth(14)
    plt.tight_layout()
    plot_filename = "clustering_comparison.pdf"
    plt.savefig(plot_filename)
    print(f"Saved plots to {plot_filename}")
