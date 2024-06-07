import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_local_results(ax, results, color, linestyle='solid'):
    means = results['client_acc_mean']
    stds = results['client_acc_std']
    xpoints = np.arange(len(means))
    ax.plot(xpoints, means, label="local", color=color, linestyle=linestyle)
    ax.fill_between(xpoints, means - stds, means + stds, alpha=0.2, color=color)


def plot_global_results(ax, results, color, linestyle='solid'):
    means = results['global_acc']
    xpoints = np.arange(len(means))
    ax.plot(xpoints, means, label="global", color=color, linestyle=linestyle, marker='^', markevery=15)


def plot_max_results(ax, results, color, linestyle='solid'):
    means = results['client_acc_max']
    xpoints = np.arange(len(means))
    ax.plot(xpoints, means, label="max", color=color, linestyle=linestyle)


def plot(filename, results, topic):
    fig, axes = plt.subplots(1, 1)
    plot_local_results(axes, results, 'C0')
    if topic == "heterofl" or topic == "fjord":
        plot_global_results(axes, results, 'C1')
    # elif topic == "fjord":
    #     plot_max_results(axes, results, 'red')
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Top-1 Accuracy")
    plt.xlabel("Round")
    if topic in ["heterofl", "fjord"]:
        plt.legend(title="Model", loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=320)
    print(f"Saved to {filename}")
    plt.close(fig)


def summary(results):
    means = results['client_acc_mean'][-1]
    stds = results['client_acc_std'][-1]
    mins = results['client_acc_min'][-1]
    maxs = results['client_acc_max'][-1]
    return f"Mean: {means:.3%} Std: {stds:.3%} Min: {mins:.3%} Max: {maxs:.3%}"


def find_stopping_point(results, key='client_acc_mean'):
    prev_mean = 0
    for i, mean in enumerate(results[key]):
        if mean < prev_mean - 0.1:
            return i - 1
        prev_mean = mean
    return i


def average_results(result_fns):
    results = []
    for resultfn in result_fns:
        with open(f"results/{resultfn}", "rb") as f:
            results.append(pickle.load(f))
    final_results = {}
    for key in results[0].keys():
        if 'client' in key:
            data = np.array([r[key] for r in results])
            data = data[np.argsort(data.mean(axis=(1, 2)))][2:]
            final_results[f"{key}_mean"] = np.mean(data.mean(axis=0), axis=1)
            final_results[f"{key}_std"] = np.mean(data.std(axis=0), axis=1)
            final_results[f"{key}_min"] = np.mean(data.min(axis=0), axis=1)
            final_results[f"{key}_max"] = np.mean(data.max(axis=0), axis=1)
        else:
            final_results[key] = np.mean([r[key] for r in results], axis=0)
    return final_results


def process_key_info(key):
    algorithm = key[:key.find('C') - 1]
    key = key[key.find('C') + 1:]
    clients = key[:key.find('_')]
    key = key[key.find('_') + 1:]
    dataset = key[:re.search(r'(_r\d)|\.', key).start()]
    key = key[re.search(r'(_r\d)|\.', key).start() + 1:]
    info = key[:key.find('.')]
    return {'algorithm': algorithm, 'clients': clients, 'dataset': dataset, 'info': info}


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})
    result_files = {}
    for resultfn in os.listdir("results"):
        key = resultfn[2:]
        if result_files.get(key) is None:
            result_files[key] = [resultfn]
        else:
            result_files[key].append(resultfn)
    all_results = []
    os.makedirs("plots", exist_ok=True)
    for key, result_files in result_files.items():
        print(key)
        results = average_results(result_files)
        plot(f"plots/{key.replace('.pkl', '')}.png", results, topic=key[:key.find('_')])
        print(summary(results))
        print()
        # sp = find_stopping_point(results)
        # sp = find_stopping_point(results, 'global_acc')
        # sp = 499
        sp = min(find_stopping_point(results), find_stopping_point(results, 'global_acc'))
        (result_data := process_key_info(key)).update({k: v[sp] for k, v in results.items()})
        result_data['stopping_point'] = sp + 1
        all_results.append(result_data)
    all_results = pd.DataFrame.from_records(all_results)
    all_results = all_results.drop(all_results[(all_results['algorithm'] == 'ours') & (all_results['dataset'].str.contains('(KD)|(counter)'))].index)
    all_results = all_results.drop(all_results[all_results['info'] == 'r5000_e1'].index)
    all_results['dataset'] = all_results['dataset'].str.replace('_pruned_((mlp)|(cnn))', '', regex=True)
    all_results['dataset'] = all_results['dataset'].str.replace('GMS-norm_scaling_', '')
    all_results['dataset'] = all_results['dataset'].str.replace('_10', '')
    all_results = all_results[['dataset', 'algorithm', 'clients', 'stopping_point', 'client_acc_mean', 'client_acc_std', 'global_acc']]
    all_results = all_results.sort_values(by=['dataset', 'algorithm', 'clients'])
    s = all_results.style.format({k: '{:,.3%}'.format for k in all_results.columns[all_results.columns.str.contains('acc')]})
    latex = s.to_latex().replace('%', r'\%').replace('_', ' ')
    print(latex)
