import polars as pl


if __name__ == "__main__":
    results = pl.read_csv("results/main.csv")
    summarised_results = pl.sql(
        """
        SELECT compressor, dataset, aggregator, adversary_type, percent_adversaries, AVG(accuracy), AVG(asr)
        FROM results
        GROUP BY dataset, compressor, aggregator, adversary_type, percent_adversaries
        ORDER BY compressor, dataset, aggregator, adversary_type, percent_adversaries
        """
    ).collect()
    print(summarised_results.to_pandas().style.hide().to_latex())
