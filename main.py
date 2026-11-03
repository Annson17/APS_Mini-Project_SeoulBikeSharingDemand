"""Runs the whole project in order.   Usage:  python main.py"""
import json, os
import numpy as np
import config
from preprocessing import load_and_clean, split_data, encode_features
from eda import run_eda
from descriptive_stats import run_descriptive
from probability import run_probability
from models import train_models, evaluate, plot_predictions, lag_baseline
from sklearn.model_selection import TimeSeriesSplit
from inference import run_data_inference, compare_models, clt_demo


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    rng = np.random.default_rng(config.SEED)
    results = {}

    section("1. PREPROCESSING")
    df, report = load_and_clean()
    print(report)
    results["preprocessing"] = report

    section("2. EDA (figures saved to outputs/figures)")
    results["eda"] = run_eda(df)

    section("3. DESCRIPTIVE STATISTICS")
    results["descriptive"] = run_descriptive(df)

    section("4. PROBABILITY")
    results["probability"] = run_probability(df)

    section("5. INFERENTIAL STATISTICS ON THE DATA")
    results["data_inference"] = run_data_inference(df)

    section("6. MODEL TRAINING AND EVALUATION (regression: predict bikes per hour)")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"train={len(X_train)}, test={len(X_test)}, target = hourly bike count "
          f"(test mean {y_test.mean():.0f}, SD {y_test.std():.0f})")
    baseline, improved = train_models(X_train, y_train)
    _, m_lag = evaluate(lambda x: lag_baseline(x), X_test, y_test, "Baseline (Prev hour)")
    pred_a, m_a = evaluate(baseline, X_test, y_test, "Baseline (LinearReg)")
    pred_b, m_b = evaluate(improved, X_test, y_test, "Improved (RandForest)")
    plot_predictions(y_test, pred_a, pred_b)
    folds = []
    cv = TimeSeriesSplit(n_splits=5)
    ordered = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
    encoded = encode_features(ordered).dropna(
        subset=["Previous hour count", "Previous day count"]
    )
    y_all = encoded.pop("Count").to_numpy(dtype=float)
    for fold, (tr, te) in enumerate(cv.split(encoded), 1):
        fold_a, fold_b = train_models(encoded.iloc[tr], y_all[tr])
        pa = fold_a.predict(encoded.iloc[te])
        pb = fold_b.predict(encoded.iloc[te])
        folds.append({"fold": fold, "n_train": len(tr), "n_test": len(te),
                      "linear_regression_rmse": float(np.sqrt(np.mean((y_all[te] - pa) ** 2))),
                      "random_forest_rmse": float(np.sqrt(np.mean((y_all[te] - pb) ** 2)))})
    results["models"] = {"split": "chronological", "n_train": len(X_train), "n_test": len(X_test),
                         "test_mean": float(y_test.mean()), "test_sd": float(y_test.std()),
                         "previous_hour": m_lag,
                         "baseline": m_a, "improved": m_b}
    results["time_series_cv"] = {
        "folds": folds,
        "linear_regression_rmse_mean": float(np.mean([x["linear_regression_rmse"] for x in folds])),
        "linear_regression_rmse_sd": float(np.std([x["linear_regression_rmse"] for x in folds], ddof=1)),
        "random_forest_rmse_mean": float(np.mean([x["random_forest_rmse"] for x in folds])),
        "random_forest_rmse_sd": float(np.std([x["random_forest_rmse"] for x in folds], ddof=1)),
    }

    section("7. SAMPLING, CONFIDENCE INTERVALS AND HYPOTHESIS TEST (model comparison)")
    results["model_comparison"] = compare_models(y_test, pred_a, pred_b, rng)

    section("8. CENTRAL LIMIT THEOREM DEMO")
    d = (y_test - pred_a) ** 2 - (y_test - pred_b) ** 2     # per-hour difference used by the paired t-test
    results["clt"] = {"rentals": clt_demo(df["Count"], "hourly rentals", "fig_clt_rentals.png", rng),
                      "sq_error_diff": clt_demo(d, "squared-error difference", "fig_clt_errors.png", rng)}

    with open(os.path.join(config.OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nDone. Results saved to outputs/results.json")


if __name__ == "__main__":
    main()
