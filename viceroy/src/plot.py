import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_performance(data_filename, plot_filename):
    df = pl.read_csv(data_filename)
    fig, ax = plt.subplots(1, 1)
    ax.plot(df["accuracy"], '-o', markevery=250, label="Accuracy", zorder=2)
    ax.plot(df["asr"], '-s', markevery=250, label="ASR", zorder=1)
    # ax.fill_between(
    #     np.arange(len(df["toggle_state"])),
    #     1.0,
    #     where=df["toggle_state"],
    #     facecolor='red',
    #     alpha=0.25,
    # )
    ax.set_ylim((-0.05, 1.05))
    ax.set_xlabel("Round")
    ax.set_ylabel("Rate")
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([
        box.x0, box.y0 + box.height * 0.1,
        box.width, box.height * 0.9,
    ])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    plt.savefig(plot_filename)
    return f"Saved plot to {plot_filename}"


def plot_influence(data_filename, plot_filename):
    df = pl.read_csv(data_filename)
    plt.violinplot([
        df.filter(pl.col("start_on")).select("p").to_numpy().reshape(-1),
        df.filter(~pl.col("start_on")).select("p").to_numpy().reshape(-1),
    ])
    plt.ylabel("Influence ($p$)")
    plt.ylim((-0.05, 1.05))
    plt.xlabel("Starting state")
    plt.xticks([1, 2], ["On", "Off"])
    plt.savefig(plot_filename)
    return f"Saved plot to {plot_filename}"



if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    # print(plot_performance("results/performance.csv", "plots/performance.pdf"))
    print(plot_influence("results/influence.csv", "plots/influence.pdf"))
