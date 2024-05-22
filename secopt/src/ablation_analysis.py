import argparse
from typing import Iterable
import logging
import itertools
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt


MARKERS = itertools.cycle(['*', 'x', '^', 'v', 'o', 's'])


def limited_query(df, **kwargs):
    return df.query(' and '.join([f"`{k}` == '{v}'" for k, v in kwargs.items()]))


def analysis_results(df, return_dict=False, **kwargs):
    exp_variables = []
    k_to_remove = []
    for k, v in kwargs.items():
        if isinstance(v, Iterable) and not isinstance(v, str):
            exp_variables.append([{k: sub_v} for sub_v in v])
            k_to_remove.append(k)
    for k in k_to_remove:
        del kwargs[k]
    results = {} if return_dict else []
    for ev in itertools.product(*exp_variables):
        exp_consts = kwargs.copy()
        for v in ev:
            exp_consts.update(v)
        filtered_results = limited_query(df, attack=args.attack, **exp_consts)
        if len(filtered_results) > 0:
            if return_dict:
                key = ' '.join([' '.join(v.values()) for v in ev])
                results[key] = filtered_results
            else:
                results.append(filtered_results)
                logging.info(f"Summary statistics for: {ev}")
                logging.info(results[-1].describe())
    if len(results) > 1 and not return_dict:
        print(f"ANOVA results: {sps.f_oneway(*[r.ssim for r in results])}")
        concat_results = pd.concat(results)
        print("Pearson correlation between the accuracy and attack SSIM: {}".format(
            sps.pearsonr(concat_results.accuracy, concat_results.ssim)
        ))
    return results


def plot_results(results, label):
    ssim_means = [r.ssim.mean() for r in results]
    acc_means = [r.accuracy.mean() for r in results]
    plt.scatter(acc_means, ssim_means, label=label, marker=next(MARKERS))


def process_labels(labels):
    result_labels = []
    for label in labels:
        label = label.replace("_", " ")
        label = label.title()
        label = label.replace("Relu", "ReLU")
        label = label.replace("Elu", "ELU")
        label = label.replace("None", "")
        label = label.strip()
        if label == "":
            label = "Baseline"
        result_labels.append(label)
    return result_labels


def violin_plot(results, title):
    plt.violinplot([v.ssim for v in results.values()], showmeans=True)
    plt.ylabel("SSIM")
    processed_labels = process_labels(results.keys())
    plt.xticks(np.arange(len(results.keys())) + 1, labels=processed_labels)
    plt.ylim((min(0.0, min([v.ssim.min() for v in results.values()]) - 0.05), 1.0))
    plt.savefig(f"{title}.png", dpi=320)
    plt.clf()


def unique_not_none(X):
    return [x for x in pd.unique(X) if x != "none"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse the results from the ablation experiments.")
    parser.add_argument("-a", "--attack", type=str, default="representation", help="The attack data to analyse.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Whether to print all information.")
    parser.add_argument("-p", "--plot", action="store_true", help="Create a plot of the results.")
    args = parser.parse_args()
    logging.basicConfig(
        format='[%(levelname)s %(asctime)s] %(filename)s:line %(lineno)d\n%(message)s',
        level=logging.INFO if args.verbose else logging.WARNING
    )

    df = pd.read_csv("results/ablation_results.csv").dropna()
    print(f"Pearson correlation between the accuracy and attack SSIM: {sps.pearsonr(df.accuracy, df.ssim)}")
    print(f"Pearson correlation between the accuracy and attack PSNR: {sps.pearsonr(df.accuracy, df.psnr)}")
    print(f"Pearson correlation between the attack SSIM and attack PSNR: {sps.pearsonr(df.ssim, df.psnr)}")

    if args.plot:
        act_results = analysis_results(
            df, return_dict=True, activation=unique_not_none(df.activation), pooling="none", normalisation="none"
        )
        violin_plot(act_results, "Activations")

        e30_df = pd.read_csv("results/e30_ablation_results.csv").dropna()
        e30_act_results = analysis_results(
            e30_df, return_dict=True, activation=unique_not_none(e30_df.activation), pooling="none", normalisation="none"
        )
        violin_plot(e30_act_results, "30 Epoch Activations")

        apn_results = analysis_results(
            df, return_dict=True, activation="relu", pooling=["none", "max_pool"], normalisation=["none", "LayerNorm"]
        )
        violin_plot(apn_results, "APN")

        bs1_df = pd.read_csv("results/bs1_ablation_results.csv").dropna()
        rp_df = pd.read_csv("results/rp_bs1_ablation_results.csv").dropna()
        sc_df = pd.read_csv("results/sc_bs1_ablation_results.csv").dropna()
        full_results = {
            "Batch size 8": df,
            "Batch size 1": bs1_df,
            "Repeated pattern": rp_df,
            "Solid colour": sc_df,
        }
        violin_plot(full_results, "Full")
        exit(0)

    print("Activation analysis")
    results = analysis_results(df, activation=unique_not_none(df.activation), pooling="none", normalisation="none")
    print()
    print("Pooling analysis")
    results = analysis_results(
        df, activation="relu", pooling=unique_not_none(df.pooling), pool_size="small", normalisation="none"
    )
    print()
    print("Pooling with window size analysis")
    results = analysis_results(
        df,
        activation="relu",
        pooling=unique_not_none(df.pooling),
        pool_size=unique_not_none(df.pool_size),
        normalisation="none"
    )
    print()
    print("Normalisation analysis")
    results = analysis_results(df, activation="relu", pooling="none", normalisation=unique_not_none(df.normalisation))
    if args.plot:
        plot_results(results, "norm")
    print()
    print("Activation + pooling analysis")
    results = analysis_results(
        df,
        activation=unique_not_none(df.activation),
        pooling=unique_not_none(df.pooling),
        pool_size=unique_not_none(df.pool_size),
        normalisation="none"
    )
    print()
    print("Activation + normalisation analysis")
    results = analysis_results(
        df,
        activation=unique_not_none(df.activation),
        pooling="none",
        pool_size="small",
        normalisation=unique_not_none(df.normalisation)
    )
    print()
    print("Pooling + normalisation analysis")
    results = analysis_results(
        df,
        activation="relu",
        pooling=unique_not_none(df.pooling),
        pool_size=unique_not_none(df.pool_size),
        normalisation=unique_not_none(df.normalisation)
    )
    print()
    print("All analysis")
    results = analysis_results(
        df,
        activation=unique_not_none(df.activation),
        pooling=unique_not_none(df.pooling),
        pool_size=unique_not_none(df.pool_size),
        normalisation=unique_not_none(df.normalisation)
    )
    print()
