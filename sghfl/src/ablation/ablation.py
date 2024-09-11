import argparse
import time
import os
import numpy as np
import scipy as sp
from tqdm import trange

import aggregators


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSFGM synthetic testbed program ablating the algorithm."
    )
    parser.add_argument('-s', '--seed', type=int, default=14258, help="Initial seed for the experiments.")
    parser.add_argument('-r', '--repetitions', type=int, default=1000,
                        help="Number of times to repeat the experiment")
    parser.add_argument('-a', '--attack', type=str, default="shifted_random", help="Type of attack to perform.")
    parser.add_argument("--space-sampling", action="store_true")
    parser.add_argument("--fractional-geomedian", action="store_true")
    parser.add_argument("--padversaries", type=float, default=0.4, help="Proportion of adversaries.")
    parser.add_argument("--aggregator", type=str, default="ssfgm", help="Aggregation algorithm to evaluate.")
    args = parser.parse_args()
    print(f"Experiment args: {vars(args)}")
    npoints = 1000
    dimensions = 2
    attack = args.attack if args.padversaries > 0.0 else "no_attack"

    start_time = time.time()
    rng = np.random.default_rng(args.seed)
    nadversaries = round(npoints * args.padversaries)

    errors = np.zeros(args.repetitions)
    improvements = np.zeros(args.repetitions)
    for r in (pbar := trange(args.repetitions)):
        honest_x = rng.normal(1, 3, size=(npoints - nadversaries, dimensions))
        match attack:
            case "lie":
                s = npoints // 2 + 1 - nadversaries
                zmax = sp.stats.norm.ppf((npoints - s) / npoints)
                attack_x = np.tile(np.mean(honest_x, 0) + zmax * np.std(honest_x, 0), (nadversaries, 1))
                x = np.concatenate((honest_x, attack_x))
            case "shifted_random":
                attack_x = rng.normal(6, np.std(honest_x, 0), (nadversaries, dimensions))
                x = np.concatenate((honest_x, attack_x))
            case "no_attack":
                additional_x = rng.normal(1, 3, size=(nadversaries, dimensions))
                honest_x = np.concatenate((honest_x, additional_x))
                x = honest_x
        if args.aggregator == "ssfgm":
            agg_mean = aggregators.ssfgm(
                x,
                space_sampling=args.space_sampling,
                fractional_geomedian=args.fractional_geomedian,
            )
        else:
            agg_mean = getattr(aggregators, args.aggregator)(x)
        honest_mean = honest_x.mean(0)
        full_mean = x.mean(0)
        errors[r] = np.linalg.norm(honest_mean - agg_mean)
        if attack == "no_attack":
            improvements[r] = 0.0
        else:
            improvements[r] = 1 - errors[r] / np.linalg.norm(honest_mean - full_mean)
        pbar.set_postfix_str(f"ERR: {errors[r]:.3f}, IMP: {improvements[r]:.3%}")

    print(("=" * 20) + " Results " + ("=" * 20))
    print(f"Average Error: {errors.mean()}, STD Error: {errors.std()}")
    print(f"Average Improvement: {improvements.mean():%}, STD Improvement: {improvements.std():%}")
    print("=" * (40 + 9))

    # Save results to csv
    os.makedirs("results", exist_ok=True)
    results_fn = "results/ablation.csv"
    experiment_results = vars(args)  # Include the information
    experiment_results["error mean"] = errors.mean()
    experiment_results["error std"] = errors.std()
    experiment_results["improvement mean"] = improvements.mean()
    experiment_results["improvement std"] = improvements.std()
    if not os.path.exists(results_fn):
        with open(results_fn, "w") as f:
            f.write(",".join(experiment_results.keys()) + "\n")
    with open(results_fn, 'a') as f:
        f.write(",".join([str(v) for v in experiment_results.values()]) + "\n")
    print(f"Results written to {results_fn}")

    print(f"Experiment took {time.time() - start_time} seconds")
