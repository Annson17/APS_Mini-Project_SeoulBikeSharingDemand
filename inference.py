"""
Step 6 - Inferential statistics.
Part A: general inference on the bike data (CI for a mean, t-test, ANOVA, chi-square, correlation).
Part B: comparing the two regression models on the same test set (paired bootstrap
        percentile CIs for RMSE and the RMSE improvement, paired t-test on squared
        errors, Wilcoxon signed-rank check, sampling-distribution experiment)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import config


# =============================== PART A ===============================
def run_data_inference(df):
    res = {}
    c = df["Count"]

    # 1. 95% confidence interval for the MEAN hourly rentals (t-interval)
    lo, hi = stats.t.interval(0.95, df=len(c) - 1, loc=c.mean(), scale=stats.sem(c))
    res["mean_ci"] = [float(lo), float(hi)]
    print(f"\n95% CI for mean hourly rentals: [{lo:.1f}, {hi:.1f}]  (sample mean {c.mean():.1f})")

    # 2. Welch two-sample t-test: do rainy hours have different mean rentals?
    #    H0: mean(rain) = mean(no rain)
    rain, dry = df.loc[df["Rain"] > 0, "Count"], df.loc[df["Rain"] == 0, "Count"]
    t, p = stats.ttest_ind(rain, dry, equal_var=False)
    res["welch_rain"] = {"mean_rain": float(rain.mean()), "mean_dry": float(dry.mean()),
                         "t": float(t), "p": float(p), "n_rain": len(rain), "n_dry": len(dry)}
    print(f"Welch t-test rain vs dry: means {rain.mean():.0f} vs {dry.mean():.0f}, t={t:.2f}, p={p:.3g}")

    # 3. One-way ANOVA: does mean demand differ across the 4 seasons?
    groups = [g["Count"].values for _, g in df.groupby("Season")]
    F, p = stats.f_oneway(*groups)
    res["anova_season"] = {"F": float(F), "p": float(p)}
    print(f"ANOVA across seasons: F={F:.1f}, p={p:.3g}")

    # 4. Chi-square test of independence: is 'high demand' independent of 'rain'?
    high = c > c.median()
    table = pd.crosstab(high, df["Rain"] > 0)
    chi2, p, dof, _ = stats.chi2_contingency(table)
    res["chi2_rain_high"] = {"chi2": float(chi2), "p": float(p), "dof": int(dof)}
    print(f"Chi-square (High demand vs Rain): chi2={chi2:.1f}, dof={dof}, p={p:.3g}")

    # 5. Pearson correlation test: Temperature vs rentals
    r, p = stats.pearsonr(df["Temp"], c)
    res["pearson_temp"] = {"r": float(r), "p": float(p)}
    print(f"Pearson correlation Temp vs Count: r={r:.3f}, p={p:.3g}")
    return res


# =============================== PART B ===============================
def rmse(sq_errors):
    return float(np.sqrt(np.mean(sq_errors)))


def compare_models(y_test, pred_a, pred_b, rng):
    """
    Is the improved model (B) really better than the baseline (A)?
    Each test hour gives one squared error per model:  e = (actual - predicted)^2.
    RMSE = sqrt(mean of squared errors).  Because both models predict the SAME hours,
    the errors are PAIRED, so we test the per-hour differences.
    """
    n = len(y_test)
    se_a, se_b = (y_test - pred_a) ** 2, (y_test - pred_b) ** 2
    ae_a, ae_b = np.abs(y_test - pred_a), np.abs(y_test - pred_b)
    rmse_a, rmse_b = rmse(se_a), rmse(se_b)
    diff = rmse_a - rmse_b                       # positive = improved model has smaller error
    print(f"\nBaseline RMSE {rmse_a:.2f}   Improved RMSE {rmse_b:.2f}   "
          f"Observed improvement (A - B) = {diff:+.2f} bikes/hour")

    # ---- Bootstrap: resample TEST ROWS with replacement, recompute RMSEs ----
    idx = rng.integers(0, n, size=(config.N_BOOT, n))
    boot_a = np.sqrt(se_a[idx].mean(axis=1))
    boot_b = np.sqrt(se_b[idx].mean(axis=1))
    boot_d = boot_a - boot_b
    ci_a, ci_b = np.percentile(boot_a, [2.5, 97.5]), np.percentile(boot_b, [2.5, 97.5])
    ci_d = np.percentile(boot_d, [2.5, 97.5])
    print(f"Bootstrap 95% CI  baseline RMSE [{ci_a[0]:.1f}, {ci_a[1]:.1f}]   "
          f"improved RMSE [{ci_b[0]:.1f}, {ci_b[1]:.1f}]")
    print(f"Bootstrap 95% CI for RMSE improvement: [{ci_d[0]:+.2f}, {ci_d[1]:+.2f}]  SE={boot_d.std():.2f}")

    # ---- Paired t-test on squared errors ----
    # H0: mean(squared error A) = mean(squared error B)   H1: they differ
    d = se_a - se_b                              # per-hour difference in squared error
    t, p_two = stats.ttest_rel(se_a, se_b)
    p_one = stats.ttest_rel(se_a, se_b, alternative="greater").pvalue   # H1: A worse than B
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=d.mean(), scale=stats.sem(d))
    dz = d.mean() / d.std(ddof=1)                # effect size (Cohen's d for paired data)
    print(f"Paired t-test (squared errors): mean diff={d.mean():.0f}, t={t:.2f}, "
          f"two-sided p={p_two:.3g}, one-sided p={p_one:.3g}, Cohen's dz={dz:.2f}")
    print(f"  95% t-interval for mean squared-error difference: [{lo:.0f}, {hi:.0f}]")

    # ---- Robustness: Wilcoxon signed-rank on absolute errors (no normality assumption) ----
    w, p_w = stats.wilcoxon(ae_a, ae_b, alternative="two-sided")
    print(f"Wilcoxon signed-rank (absolute errors): MAE {ae_a.mean():.1f} vs {ae_b.mean():.1f}, p={p_w:.3g}")
    better_share = float(np.mean(se_b < se_a))
    print(f"Improved model has the smaller error on {100*better_share:.1f}% of test hours")

    decision = "REJECT H0" if p_two < config.ALPHA else "FAIL TO REJECT H0"
    print(f"Decision at alpha={config.ALPHA}: {decision}")

    # ---- Sampling distribution: how noisy is a SMALL test sample? ----
    demo, demo_res = {}, {}
    print("\nSampling distribution of the RMSE improvement (A - B):")
    for s_ in [50, 100, 300, 1000]:
        dd = np.empty(2000)
        for i in range(2000):
            j = rng.choice(n, size=s_, replace=False)
            dd[i] = rmse(se_a[j]) - rmse(se_b[j])
        demo[s_] = dd
        demo_res[str(s_)] = {"mean": float(dd.mean()), "sd": float(dd.std()),
                             "p_not_better": float(np.mean(dd <= 0))}
        print(f"  n={s_:5d}: mean={dd.mean():+.2f}  SD={dd.std():.2f}  P(not better)={np.mean(dd <= 0):.3f}")

    # ---- Figures ----
    plt.figure(figsize=(6, 4))
    plt.hist(boot_d, bins=50, color="#4C72B0", edgecolor="white")
    plt.axvline(0, color="red", ls="--", label="No difference (H0)")
    plt.axvline(diff, color="black", label="Observed")
    plt.axvline(ci_d[0], color="gray", ls=":"); plt.axvline(ci_d[1], color="gray", ls=":", label="95% CI")
    plt.xlabel("RMSE improvement (Baseline - Improved), bikes/hour"); plt.ylabel("Frequency")
    plt.title("Bootstrap distribution of the RMSE improvement"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig_bootstrap.png"), dpi=180); plt.close()

    plt.figure(figsize=(6, 4))
    for s_, dd in demo.items():
        plt.hist(dd, bins=40, alpha=0.5, label=f"n = {s_}")
    plt.axvline(0, color="red", ls="--")
    plt.xlabel("RMSE improvement on a random test sample"); plt.ylabel("Frequency")
    plt.title("Sampling distribution vs test-set size"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig_sampling.png"), dpi=180); plt.close()

    return {"rmse_baseline": rmse_a, "rmse_improved": rmse_b, "rmse_diff": float(diff),
            "ci_rmse_baseline": [float(x) for x in ci_a], "ci_rmse_improved": [float(x) for x in ci_b],
            "boot_ci_diff": [float(x) for x in ci_d], "boot_se": float(boot_d.std()),
            "mae_baseline": float(ae_a.mean()), "mae_improved": float(ae_b.mean()),
            "ttest": {"mean_diff_sq": float(d.mean()), "t": float(t), "p_two": float(p_two),
                      "p_one": float(p_one), "ci": [float(lo), float(hi)], "dz": float(dz)},
            "wilcoxon": {"stat": float(w), "p": float(p_w)},
            "improved_better_share": better_share,
            "decision": decision, "sampling_demo": demo_res}


# =============================== PART C ===============================
def clt_demo(values, label, fig_name, rng, sizes=(1, 5, 30, 200), n_rep=5000):
    """
    Central Limit Theorem demo. 'values' is a skewed population (e.g. hourly rentals, or the
    per-hour squared-error differences used in the paired t-test). For each sample size n we
    draw n_rep random samples, take each sample's MEAN, and look at the distribution of means.
    CLT: it becomes bell-shaped, centred at mu, with SD = sigma / sqrt(n) (the standard error).
    """
    values = np.asarray(values, dtype=float)
    mu, sigma = values.mean(), values.std()
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.2 * len(sizes), 3.4), sharey=False)
    out = {}
    for ax, n in zip(axes, sizes):
        means = np.array([values[rng.integers(0, len(values), n)].mean() for _ in range(n_rep)])
        ax.hist(means, bins=40, density=True, color="#4C72B0", edgecolor="white")
        x = np.linspace(means.min(), means.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma / np.sqrt(n)), color="red", lw=1.5)
        ax.set_title(f"n = {n}\nskew = {stats.skew(means):.2f}")
        ax.set_xlabel("sample mean")
        out[str(n)] = {"mean_of_means": float(means.mean()), "sd_of_means": float(means.std()),
                       "theory_se": float(sigma / np.sqrt(n)), "skew": float(stats.skew(means))}
    fig.suptitle(f"CLT: sampling distribution of the mean - {label}  "
                 f"(population skew = {stats.skew(values):.2f}; red = Normal(mu, sigma/sqrt(n)))", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, fig_name), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nCLT demo ({label}): population mean={mu:.1f}, SD={sigma:.1f}, skew={stats.skew(values):.2f}")
    for n, r in out.items():
        print(f"  n={n:>4}: SD of sample means={r['sd_of_means']:.2f}  vs sigma/sqrt(n)={r['theory_se']:.2f}  "
              f"skew={r['skew']:.2f}")
    return out
