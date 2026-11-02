"""Runs the whole project in order.   Usage:  python main.py"""
import json, os
import numpy as np
import config
from preprocessing import load_and_clean, split_data
from eda import run_eda
from descriptive_stats import run_descriptive
from probability import run_probability
from models import train_models, evaluate, plot_predictions
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
    pred_a, m_a = evaluate(baseline, X_test, y_test, "Baseline (LinearReg)")
    pred_b, m_b = evaluate(improved, X_test, y_test, "Improved (RandForest)")
    plot_predictions(y_test, pred_a, pred_b)
    results["models"] = {"n_train": len(X_train), "n_test": len(X_test),
                         "test_mean": float(y_test.mean()), "test_sd": float(y_test.std()),
                         "baseline": m_a, "improved": m_b}

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
