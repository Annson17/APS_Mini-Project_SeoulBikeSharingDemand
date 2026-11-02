"""Step 3 - Descriptive statistics: summarise the data with numbers."""
import numpy as np
import pandas as pd
from scipy import stats


def run_descriptive(df):
    c = df["Count"]
    out = {
        "mean": c.mean(), "median": c.median(), "mode": c.mode().iloc[0],
        "std": c.std(), "variance": c.var(),
        "min": c.min(), "max": c.max(),
        "q1": c.quantile(0.25), "q3": c.quantile(0.75),
        "iqr": c.quantile(0.75) - c.quantile(0.25),
        "skewness": stats.skew(c), "kurtosis": stats.kurtosis(c),
        "cv_percent": 100 * c.std() / c.mean(),      # coefficient of variation
    }
    # Outliers by the 1.5*IQR rule
    low, high = out["q1"] - 1.5 * out["iqr"], out["q3"] + 1.5 * out["iqr"]
    out["n_outliers_iqr"] = int(((c < low) | (c > high)).sum())

    print("\n=== Descriptive statistics of hourly rentals ===")
    for k, v in out.items():
        print(f"{k:16s} {v:10.3f}")

    by_season = df.groupby("Season")["Count"].agg(["count", "mean", "median", "std"]).round(1)
    print("\nRentals by season:\n", by_season)

    num_cols = ["Temp", "Humidity", "Wind", "Visibility", "Solar", "Rain", "Snow"]
    summary = df[num_cols + ["Count"]].describe().T.round(2)
    print("\nSummary of numeric features:\n", summary)

    corr = df[num_cols + ["Hour", "Count"]].corr()["Count"].drop("Count").round(3)
    print("\nCorrelation of each feature with rentals:\n", corr)

    return {"count_summary": {k: float(v) for k, v in out.items()},
            "by_season": by_season.to_dict(orient="index"),
            "corr_with_count": corr.to_dict()}
