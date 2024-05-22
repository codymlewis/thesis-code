import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("results/inversion_results.csv")
    df = df.drop(columns=["model", "pgd", "attack", "optimiser", "seed", "batch_size", "steps", "regularise"])
    for dataset, ddf in iter(df.groupby(["dataset", "l1_reg", "l2_reg"]).mean().reset_index().groupby(["dataset"])):
        max_ssim = ddf["ssim"].max()
        print(ddf.query("`ssim` == @max_ssim"))

    df = pd.read_csv("results/precode_inversion_results.csv")
    df = df.drop(columns=["model", "pgd", "attack", "seed", "batch_size"])
    for dataset, ddf in iter(df.groupby(["dataset", "l1_reg", "l2_reg"]).mean().reset_index().groupby(["dataset"])):
        max_ssim = ddf["ssim"].max()
        print(ddf.query("`ssim` == @max_ssim"))
