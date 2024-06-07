import argparse
import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import jax


def create_plot(plot_data, save_filename):
    analytics = {k: np.array([pd[k] for pd in plot_data['analytics']]) for k in plot_data['analytics'][0].keys()}
    evaluations = np.array(plot_data['evaluation'])
    rounds = np.arange(len(evaluations)) + 1
    if "feddrop" not in save_filename:
        plt.plot(rounds, analytics['mean'], label="Local", marker='s', color="#1f77b4", markevery=5)
        plt.fill_between(rounds, analytics['min'], analytics['max'], alpha=0.2)
    if "ppdhfl" not in save_filename:
        plt.plot(rounds, evaluations, label="Global", marker='^', color="#ff7f0e", markevery=5)
    if "feddrop" not in save_filename and "ppdhfl" not in save_filename:
        plt.legend(title="Model", loc='lower right')
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Top-1 Accuracy")
    plt.xlabel("Round")
    plt.tight_layout()
    plt.savefig(save_filename, dpi=320)
    print(f"Saved to {save_filename}")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create plots from the experiments results.")
    parser.add_argument("-a", "--allocation", type=str, default="cyclic",
                        help="The allocation type to look at the results from.")
    args = parser.parse_args()

    for framework in ["ppdhfl", "feddrop", "heterofl", "fjord"]:
        plot_data_fns = [
            fn for fn in os.listdir("results")
            if re.search(f"dataset=mnist.*seed=\\d.*allocation={args.allocation}.*framework={framework}", fn)
        ]
        plot_data_collection = []
        for plot_data_fn in plot_data_fns:
            with open(f"results/{plot_data_fn}", "r") as f:
                plot_data_collection.append(json.load(f))
        create_plot(jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *plot_data_collection), f"{framework}.png")
