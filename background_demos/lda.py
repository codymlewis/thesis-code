import argparse
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to visualise the Latent Dirichlet Allocation"
    )
    parser.add_argument(
        "--num-clients",
        default=10,
        type=int,
        help="Number of clients to allocate for."
    )
    parser.add_argument(
        "--num-classes",
        default=10,
        type=int,
        help="Number of classes to allocate for."
    )
    parser.add_argument(
        "--seed",
        default=823,
        type=int,
        help="The seed for the random generation used in the the Directlet distribution."
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    for alpha in [0.0000001, 0.1, 1.0, 100, 1000000]:
        allocation_matrix = rng.dirichlet(
            np.repeat(alpha, args.num_clients),
            size=args.num_classes,
        )

        for c in range(args.num_classes):
            plt.barh(
                np.arange(args.num_clients),
                allocation_matrix[:, c],
                left=np.sum(allocation_matrix[:, :c], axis=1),
                color=f"C{c}",
            )
        plt.xticks([])
        plt.yticks([])

        filename = f"lda_alpha={alpha}.pdf"
        plt.savefig(filename)
        plt.clf()
        print(f"Saved {filename}")
