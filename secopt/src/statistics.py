import os
import pandas as pd


if __name__ == "__main__":
    inversion_fn = "results/inversion_results.csv"
    if os.path.exists(inversion_fn):
        print("Showing inversion attack results:")
        inversion_df = pd.read_csv(inversion_fn)
        inversion_df = inversion_df.groupby(
            ['dataset', 'model', 'pgd', 'attack', 'optimiser', 'batch_size', 'steps', 'regularise']
        ).mean()
        inversion_df = inversion_df.reset_index()
        print(inversion_df.style.hide().to_latex(position_float='centering'))
        print()

    performance_fn = "results/performance_results.csv"
    if os.path.exists(performance_fn):
        print("Showing performance results:")
        performance_df = pd.read_csv(performance_fn)
        performance_df = performance_df.drop(columns=[
            "seed",
            "clients",
            "rounds",
            "epochs",
            "steps",
            "batch_size",
            "participation_rate",
            "client_learning_rate",
            "server_learning_rate",
        ])
        performance_df = performance_df.groupby([
            'dataset', 'model', 'pgd', 'server_optimiser', 'client_optimiser', 'clip_threshold', 'noise_scale'
        ]).mean()
        performance_df = performance_df.reset_index()
        print(performance_df.style.hide().to_latex(position_float='centering'))
        print()

    print("=" * 20)
    print("PRECODE Results")
    print("=" * 20)

    inversion_fn = "results/precode_inversion_results.csv"
    if os.path.exists(inversion_fn):
        print("Showing precode inversion attack results:")
        inversion_df = pd.read_csv(inversion_fn)
        inversion_df = inversion_df.groupby(['dataset', 'model', 'pgd', 'attack', 'batch_size']).mean()
        inversion_df = inversion_df.reset_index()
        print(inversion_df.style.hide().to_latex(position_float='centering'))
        print()

    performance_fn = "results/precode_performance_results.csv"
    if os.path.exists(performance_fn):
        print("Showing precode performance results:")
        performance_df = pd.read_csv(performance_fn)
        performance_df = performance_df.drop(columns=[
            "seed",
            "clients",
            "rounds",
            "epochs",
            "steps",
            "batch_size",
            "learning_rate",
            "train_inversion",
            "performance",
            "attack",
            "file",
            "runs",
            "l1_reg",
            "l2_reg",
        ])
        performance_df = performance_df.groupby(['dataset', 'model', 'pgd']).mean()
        performance_df = performance_df.reset_index()
        print(performance_df.style.hide().to_latex(position_float='centering'))
        print()
