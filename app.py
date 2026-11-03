import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from preprocessing import load_and_clean, split_data, encode_features
from models import train_models, lag_baseline

st.set_page_config(
    page_title="Seoul Bike Analytics",
    page_icon="🚲",
    layout="wide",
)

sns.set_theme(style="whitegrid")
ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "outputs" / "results.json"
FIG_DIR = ROOT / "outputs" / "figures"


@st.cache_data(show_spinner="Loading and cleaning Seoul Bike data...")
def get_data():
    df, report = load_and_clean()
    return df, report


@st.cache_data
def get_saved_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Saved results file not found: {RESULTS_PATH}")
    with RESULTS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner="Preparing the train/test split...")
def get_split(_df):
    return split_data(_df)


@st.cache_resource(show_spinner="Training the models for the first time...")
def get_models(_X_train, _y_train):
    # Model objects are cached for this Streamlit session/server process.
    return train_models(_X_train, _y_train)


@st.cache_data(show_spinner="Evaluating models...")
def get_predictions(_model_a, _model_b, X_test, y_test):
    pred_a = _model_a.predict(X_test)
    pred_b = _model_b.predict(X_test)
    pred_lag = lag_baseline(X_test)
    metrics = {
        "Previous-hour baseline": {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_lag))),
            "MAE": float(mean_absolute_error(y_test, pred_lag)),
            "R²": float(r2_score(y_test, pred_lag)),
        },
        "Linear Regression": {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_a))),
            "MAE": float(mean_absolute_error(y_test, pred_a)),
            "R²": float(r2_score(y_test, pred_a)),
        },
        "Random Forest": {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_b))),
            "MAE": float(mean_absolute_error(y_test, pred_b)),
            "R²": float(r2_score(y_test, pred_b)),
        },
    }
    return pred_a, pred_b, metrics


def show_metric_table(metrics):
    st.dataframe(pd.DataFrame(metrics).T.style.format({
        "RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}"
    }), use_container_width=True)


def show_saved_figure(filename, caption):
    path = FIG_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Saved figure is unavailable: {filename}")


try:
    df, prep_report = get_data()
    saved_results = get_saved_results()
    X_train, X_test, y_train, y_test = get_split(df)
    model_a, model_b = get_models(X_train, y_train)
    pred_a, pred_b, metrics = get_predictions(model_a, model_b, X_test, y_test)
except Exception as exc:
    st.error("The project could not load its data or models.")
    st.exception(exc)
    st.info(
        "Check that app.py, the data folder, and the supporting Python modules "
        "are in the project directory."
    )
    st.stop()


with st.sidebar:
    st.title("🚲 Seoul Bike Sharing Demand")
    st.caption("Applied Probability and Statistics")
    page = st.radio(
        "Navigation",
        [
            "🏠 Project Overview",
            "📊 Exploratory Data Analysis",
            "🧠 Model Comparison",
            "🧪 Statistical Analysis",
            "🔮 Bike Rental Prediction",
        ],
    )
    st.divider()
    st.caption(f"Analysis dataset: {len(df):,} hourly records")
    st.caption(f"Modelling dataset: {len(X_train) + len(X_test):,} records")
    st.caption("Chronological split: earliest 70% for training / latest 30% for testing")


# ---------------------------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------------------------
if page == "🏠 Project Overview":
    st.title("Seoul Bike Sharing Demand")
    st.subheader("Hourly demand analysis and forecasting")
    st.write(
        "This dashboard presents the exploratory, statistical, and forecasting "
        "analysis of hourly bike rentals in Seoul."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw records", f"{prep_report['rows_raw']:,}")
    c2.metric("Analysis records", f"{prep_report['rows_clean']:,}")
    c3.metric("Training records", f"{len(X_train):,}")
    c4.metric("Test records", f"{len(X_test):,}")
    st.caption("Model metrics are from the saved batch evaluation. The analysis dataset contains all operating-day records; the modelling dataset excludes the first 24 observations without complete lag values.")

    st.markdown("### Holdout performance")
    m1, m2, m3, m4 = st.columns(4)
    saved_models = saved_results["models"]
    m1.metric("Previous-hour RMSE", f"{saved_models['previous_hour']['rmse']:.1f}")
    m2.metric("Linear Regression RMSE", f"{saved_models['baseline']['rmse']:.1f}")
    m3.metric("Random Forest RMSE", f"{saved_models['improved']['rmse']:.1f}")
    m4.metric("RF improvement vs LR", f"{saved_models['baseline']['rmse'] - saved_models['improved']['rmse']:.1f} bikes/hour")

    st.markdown("### Analysis workflow")
    st.markdown(
        """
        1. **Data preparation:** validate the source file, remove non-operating-day records, and create calendar and lag features.
        2. **Exploratory analysis:** examine demand distributions, hourly patterns, seasons, and weather variables.
        3. **Probability analysis:** calculate conditional probabilities, independence checks, total probability, and Bayes' theorem examples.
        4. **Forecast comparison:** evaluate a previous-hour benchmark, Linear Regression, and Random Forest.
        5. **Uncertainty analysis:** use time-series validation, paired model comparisons, block bootstrap intervals, and CLT simulations.
        """
    )
    st.warning(
        "Interpretation note: the final test set contains later observations than the training set. "
        "Hourly demand is temporally dependent, so the inferential results are exploratory."
    )
    st.info(
        f"Across five expanding-window validation folds, Random Forest achieved a mean RMSE of "
        f"{saved_results['time_series_cv']['random_forest_rmse_mean']:.1f} "
        f"± {saved_results['time_series_cv']['random_forest_rmse_sd']:.1f} bikes/hour."
    )
    with st.expander("Data preparation details"):
        st.json(prep_report)


# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.write("Filter the operating-day observations to examine hourly demand across time, season, and weather conditions.")

    seasons = sorted(df["Season"].dropna().unique().tolist())
    chosen_seasons = st.multiselect("Seasons", seasons, default=seasons)
    hour_range = st.slider("Hours of day", 0, 23, (0, 23))
    filtered = df[
        df["Season"].isin(chosen_seasons)
        & df["Hour"].between(hour_range[0], hour_range[1])
    ].copy()

    if filtered.empty:
        st.warning("No records match these filters.")
        st.stop()

    a, b, c = st.columns(3)
    a.metric("Filtered observations", f"{len(filtered):,}")
    b.metric("Mean hourly demand", f"{filtered['Count'].mean():.1f} bikes")
    c.metric("Median hourly demand", f"{filtered['Count'].median():.1f} bikes")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Demand distribution", "Time and seasons", "Weather and correlation",
        "Batch figures"
    ])
    with tab1:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(filtered["Count"], bins=40, kde=True, ax=ax[0])
        ax[0].set_title("Hourly demand distribution")
        sns.boxplot(data=filtered, x="Count", ax=ax[1])
        ax[1].set_title("Hourly demand boxplot")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        hourly = filtered.groupby("Hour", as_index=False)["Count"].mean()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.lineplot(data=hourly, x="Hour", y="Count", marker="o", ax=ax)
        ax.set_title("Mean hourly demand by hour")
        ax.set_xticks(range(hour_range[0], hour_range[1] + 1))
        ax.set_ylabel("Mean bikes rented")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        seasonal = filtered.groupby("Season", as_index=False)["Count"].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=seasonal, x="Season", y="Count", ax=ax)
        ax.set_title("Mean hourly demand by season")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(data=filtered.sample(min(1800, len(filtered)), random_state=42),
                        x="Temp", y="Count", hue="Season", alpha=0.55, ax=ax)
        ax.set_title("Temperature and hourly demand")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        numeric_cols = ["Count", "Hour", "Temp", "Humidity", "Wind", "Visibility",
                        "DewPoint", "Solar", "Rain", "Snow"]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(filtered[numeric_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation matrix")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("Descriptive statistics for the filtered data"):
        st.dataframe(filtered[["Count", "Hour", "Temp", "Humidity", "Wind",
                               "Visibility", "DewPoint", "Solar", "Rain", "Snow"]]
                     .describe().T, use_container_width=True)

    with tab4:
        st.write("Figures generated by the batch analysis using all 8,465 operating-day observations.")
        figure_cols = st.columns(2)
        batch_figures = [
            ("eda_count_distribution.png", "Hourly demand distribution"),
            ("eda_hourly_pattern.png", "Average hourly demand"),
            ("eda_season_boxplot.png", "Demand by season"),
            ("eda_hourly_by_season.png", "Hourly demand by season"),
            ("eda_temp_scatter.png", "Temperature and demand"),
            ("eda_correlation.png", "Correlation matrix"),
        ]
        for index, (filename, caption) in enumerate(batch_figures):
            with figure_cols[index % 2]:
                show_saved_figure(filename, caption)

        with st.expander("Overall descriptive summary"):
            summary = saved_results["descriptive"]["count_summary"]
            st.dataframe(
                pd.DataFrame([summary]).T.rename(columns={0: "Value"}),
                use_container_width=True,
            )
            st.dataframe(
                pd.DataFrame(saved_results["descriptive"]["by_season"]).T,
                use_container_width=True,
            )


# ---------------------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------------------
elif page == "🧠 Model Comparison":
    st.title("🧠 Forecast Model Comparison")
    st.write("All three forecasting approaches are evaluated on the same chronological holdout.")

    show_metric_table(metrics)
    st.markdown("### Time-series validation")
    st.caption("Each fold trains on earlier observations and evaluates on the next time segment.")
    folds = pd.DataFrame(saved_results["time_series_cv"]["folds"])
    st.dataframe(folds, hide_index=True, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(folds["fold"], folds["linear_regression_rmse"], marker="o", label="Linear Regression")
    ax.plot(folds["fold"], folds["random_forest_rmse"], marker="o", label="Random Forest")
    ax.set_xlabel("Validation fold")
    ax.set_ylabel("RMSE (bikes/hour)")
    ax.set_title("RMSE across expanding-window validation folds")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Official batch uncertainty results"):
        comparison = saved_results["model_comparison"]
        uncertainty = pd.DataFrame({
            "Linear Regression": [
                comparison["rmse_baseline"], comparison["mae_baseline"],
                comparison["ci_rmse_baseline"][0], comparison["ci_rmse_baseline"][1],
            ],
            "Random Forest": [
                comparison["rmse_improved"], comparison["mae_improved"],
                comparison["ci_rmse_improved"][0], comparison["ci_rmse_improved"][1],
            ],
        }, index=["RMSE", "MAE", "Bootstrap CI lower (RMSE)", "Bootstrap CI upper (RMSE)"])
        st.dataframe(uncertainty.style.format("{:.2f}"), use_container_width=True)
        st.write(
            f"Block-bootstrap 95% CI for RMSE improvement: "
            f"[{comparison['boot_ci_diff'][0]:.2f}, {comparison['boot_ci_diff'][1]:.2f}] bikes/hour."
        )
        st.write(
            f"Random Forest had the lower squared error on "
            f"{comparison['improved_better_share'] * 100:.1f}% of holdout observations."
        )

    st.markdown("### Random Forest feature importance")
    importance = pd.Series(model_b.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    importance.plot.barh(ax=ax, color="#55A868")
    ax.set_xlabel("Impurity-based importance")
    ax.set_title("Top Random Forest predictors")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Performance metrics")
    metric_choice = st.selectbox("Select a metric", ["RMSE", "MAE", "R²"])
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(metrics.keys())
    values = [metrics[n][metric_choice] for n in names]
    ax.bar(names, values)
    ax.set_ylabel(metric_choice)
    ax.set_title(f"{metric_choice}: lower is better" if metric_choice != "R²" else "R²: higher is better")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Actual versus predicted demand")
    st.caption("The plots below show the two trained models on a sample of the chronological test observations.")
    n_points = st.slider("Test observations to display", 100, len(y_test), min(800, len(y_test)), step=100)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=n_points, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, pred, name in [
        (axes[0], pred_a, "Linear Regression"),
        (axes[1], pred_b, "Random Forest"),
    ]:
        ax.scatter(np.asarray(y_test)[idx], pred[idx], s=12, alpha=0.45)
        lim = [0, max(float(np.max(y_test)), float(np.max(pred)))]
        ax.plot(lim, lim, linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Actual bikes rented")
        ax.set_ylabel("Predicted bikes rented")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Prediction error distributions")
    errors = pd.DataFrame({
        "Linear Regression": np.asarray(y_test) - pred_a,
        "Random Forest": np.asarray(y_test) - pred_b,
    })
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(errors["Linear Regression"], color="#4C72B0", stat="density",
                 element="step", fill=False, label="Linear Regression", ax=ax)
    sns.histplot(errors["Random Forest"], color="#55A868", stat="density",
                 element="step", fill=False, label="Random Forest", ax=ax)
    ax.set_title("Prediction error (actual − predicted bikes)")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.info(
        "RMSE penalises large errors more strongly, MAE is the average absolute error, "
        "and R² measures explained variation relative to predicting the test-set mean."
    )


# ---------------------------------------------------------------------
# STATISTICAL ANALYSIS
# ---------------------------------------------------------------------
elif page == "🧪 Statistical Analysis":
    st.title("🧪 Statistical Analysis")
    st.write("Review the statistical summaries and model-comparison evidence used in this project.")
    tab1, tab2, tab3 = st.tabs(["Model comparison", "Dataset inference", "Probability and CLT"])

    with tab1:
        st.subheader("Paired comparison: Linear Regression vs Random Forest")
        st.latex(r"d_i=(y_i-\hat y_{LR,i})^2-(y_i-\hat y_{RF,i})^2")
        st.write("Null hypothesis: the mean paired squared-error difference is zero.")
        st.write("Alternative hypothesis: the mean paired squared-error difference is not zero.")
        se_a = (np.asarray(y_test) - pred_a) ** 2
        se_b = (np.asarray(y_test) - pred_b) ** 2
        d = se_a - se_b
        t_stat, p_value = stats.ttest_rel(se_a, se_b)
        alpha = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.01)
        ci_low, ci_high = stats.t.interval(
            1 - alpha, df=len(d) - 1, loc=float(np.mean(d)),
            scale=stats.sem(d)
        )
        k1, k2, k3 = st.columns(3)
        k1.metric("t-statistic", f"{t_stat:.3f}")
        k2.metric("p-value", f"{p_value:.3e}")
        k3.metric("Mean squared-error difference", f"{np.mean(d):,.0f}")
        st.write(f"{100*(1-alpha):.0f}% t confidence interval: **[{ci_low:,.0f}, {ci_high:,.0f}]**")
        if p_value < alpha:
            st.success("Reject H₀: the mean squared errors differ significantly. The positive mean difference favours Random Forest.")
        else:
            st.warning("Fail to reject H₀: the test does not provide sufficient evidence of a difference.")
        st.caption("Interpretation note: the paired t-test is based on hourly differences, although neighbouring hourly observations may be dependent.")

        st.subheader("Interactive block-bootstrap confidence interval")
        st.caption(
            "The dashboard resamples contiguous 24-observation blocks to retain "
            "some within-day dependence. The official batch analysis uses 10,000 resamples."
        )
        reps = st.slider("Bootstrap resamples", 500, 5000, 2000, step=500)
        rng = np.random.default_rng(42)
        n = len(y_test)
        improvements = np.empty(reps)
        block_ids = np.arange(n) // 24
        blocks = np.unique(block_ids)
        chunk_size = 200
        for start in range(0, reps, chunk_size):
            stop = min(start + chunk_size, reps)
            for i in range(start, stop):
                chosen = rng.choice(blocks, size=len(blocks), replace=True)
                indices = np.concatenate([
                    np.flatnonzero(block_ids == block) for block in chosen
                ])
                improvements[i] = np.sqrt(se_a[indices].mean()) - np.sqrt(se_b[indices].mean())
        boot_lo, boot_hi = np.percentile(improvements, [2.5, 97.5])
        observed = np.sqrt(se_a.mean()) - np.sqrt(se_b.mean())
        b1, b2, b3 = st.columns(3)
        b1.metric("Observed RMSE improvement", f"{observed:.2f}")
        b2.metric("95% bootstrap lower", f"{boot_lo:.2f}")
        b3.metric("95% bootstrap upper", f"{boot_hi:.2f}")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(improvements, bins=40, edgecolor="white")
        ax.axvline(0, linestyle="--", label="No improvement (0)")
        ax.axvline(observed, linestyle="-", label="Observed improvement")
        ax.axvline(boot_lo, linestyle=":", label="95% interval")
        ax.axvline(boot_hi, linestyle=":")
        ax.set_xlabel("RMSE improvement (Linear Regression − Random Forest)")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if boot_lo > 0:
            st.success("The interval is entirely above zero, indicating lower Random Forest RMSE under this block-resampling procedure.")

        with st.expander("Wilcoxon signed-rank test (absolute errors)"):
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                np.abs(np.asarray(y_test) - pred_a),
                np.abs(np.asarray(y_test) - pred_b),
            )
            st.write(f"Statistic: **{wilcoxon_stat:,.1f}**")
            st.write(f"p-value: **{wilcoxon_p:.3e}**")
            st.caption("This paired non-parametric comparison uses absolute errors. Because the data are temporal, the p-value is exploratory.")

    with tab2:
        st.subheader("Confidence interval for mean hourly demand")
        c = df["Count"].astype(float)
        lo, hi = stats.t.interval(0.95, df=len(c)-1, loc=c.mean(), scale=stats.sem(c))
        st.write(f"Mean hourly demand: **{c.mean():.2f} bikes**")
        st.write(f"95% t confidence interval: **[{lo:.2f}, {hi:.2f}]**")

        st.subheader("Welch's t-test: rainy versus dry hours")
        rain = df.loc[df["Rain"] > 0, "Count"]
        dry = df.loc[df["Rain"] == 0, "Count"]
        t_rain, p_rain = stats.ttest_ind(rain, dry, equal_var=False)
        r1, r2, r3 = st.columns(3)
        r1.metric("Mean on rainy hours", f"{rain.mean():.1f}")
        r2.metric("Mean on dry hours", f"{dry.mean():.1f}")
        r3.metric("p-value", f"{p_rain:.3e}")
        st.caption(f"Welch t-statistic = {t_rain:.2f}; rainy n = {len(rain):,}, dry n = {len(dry):,}.")

        st.subheader("One-way ANOVA: hourly demand across seasons")
        groups = [g["Count"].to_numpy() for _, g in df.groupby("Season")]
        f_stat, p_anova = stats.f_oneway(*groups)
        st.write(f"F-statistic: **{f_stat:.2f}** · p-value: **{p_anova:.3e}**")
        st.caption("A significant result indicates that at least one seasonal mean differs; it does not identify which pairs differ.")

        st.subheader("Chi-square test: high demand and rainfall")
        high = df["Count"] > df["Count"].median()
        table = pd.crosstab(high, df["Rain"] > 0)
        chi2, p_chi, dof, expected = stats.chi2_contingency(table)
        st.write(f"χ² = **{chi2:.2f}**, degrees of freedom = **{dof}**, p-value = **{p_chi:.3e}**")
        st.dataframe(table.rename(index={False: "Not high", True: "High"},
                                  columns={False: "No rain", True: "Rain"}))

        st.subheader("Pearson correlation: temperature and hourly demand")
        r, p_corr = stats.pearsonr(df["Temp"], df["Count"])
        r1, r2 = st.columns(2)
        r1.metric("Pearson r", f"{r:.3f}")
        r2.metric("p-value", f"{p_corr:.3e}")

    with tab3:
        st.subheader("Probability examples")
        high = df["Count"] > df["Count"].median()
        rain_event = df["Rain"] > 0
        rush = df["Hour"].isin([7, 8, 9, 17, 18, 19])
        probability_table = pd.DataFrame({
            "Probability": [
                high.mean(), rain_event.mean(),
                df.loc[rain_event, "Count"].gt(df["Count"].median()).mean(),
                df.loc[~rain_event, "Count"].gt(df["Count"].median()).mean(),
                df.loc[rush, "Count"].gt(df["Count"].median()).mean(),
            ]
        }, index=["P(High)", "P(Rain)", "P(High | Rain)", "P(High | No Rain)", "P(High | Rush hour)"])
        st.dataframe(probability_table.style.format("{:.4f}"), use_container_width=True)
        st.caption("High demand means rentals above the analysis-dataset median. Rain means rainfall above 0 mm. Rush hours are 07:00–09:00 and 17:00–19:00.")
        st.subheader("Complete probability results")
        probability_results = pd.Series(
            saved_results["probability"], name="Probability"
        )
        st.dataframe(
            probability_results.to_frame().style.format("{:.4f}"),
            use_container_width=True,
        )

        st.subheader("Central Limit Theorem simulation")
        values_source = st.selectbox("Series to sample from", ["Hourly demand", "Squared-error differences"])
        values = df["Count"].to_numpy(dtype=float) if values_source == "Hourly demand" else d.astype(float)
        sample_sizes = [1, 5, 30, 200]
        n_rep = 1500
        rng = np.random.default_rng(42)
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
        clt_rows = []
        for ax, sample_n in zip(axes, sample_sizes):
            sample_means = np.array([
                values[rng.integers(0, len(values), size=sample_n)].mean()
                for _ in range(n_rep)
            ])
            ax.hist(sample_means, bins=30, density=True, edgecolor="white")
            ax.set_title(f"n={sample_n}\nskew={stats.skew(sample_means):.2f}")
            ax.set_xlabel("Sample mean")
            clt_rows.append({
                "Sample size": sample_n,
                "Mean of sample means": sample_means.mean(),
                "Observed SD": sample_means.std(),
                "Theoretical SE": values.std() / np.sqrt(sample_n),
                "Skewness": stats.skew(sample_means),
            })
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(pd.DataFrame(clt_rows).set_index("Sample size").style.format("{:.3f}"), use_container_width=True)
        st.caption("As sample size increases, the distribution of sample means generally becomes less skewed and its standard error decreases. This simulation does not remove the temporal-dependence limitation.")


# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------
elif page == "🔮 Bike Rental Prediction":
    st.title("🔮 Predict Hourly Bike Rentals")
    st.write("Enter the conditions and recent demand values required to compare the two trained forecasting models.")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            hour = st.slider("Hour of day", 0, 23, 18)
            month = st.slider("Month", 1, 12, 7)
            weekend = st.selectbox("Day type", ["Weekday", "Weekend"]) == "Weekend"
            season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
            holiday = st.selectbox("Holiday status", ["No holiday", "Holiday"]) == "Holiday"
            previous_hour = st.number_input("Previous-hour demand", min_value=0, max_value=4000, value=500)
            previous_day = st.number_input("Previous-day demand", min_value=0, max_value=4000, value=500)
            temp = st.slider("Temperature (°C)", -17.8, 39.4, 15.0)
            humidity = st.slider("Humidity (%)", 0, 100, 58)
        with right:
            wind = st.slider("Wind speed (m/s)", 0.0, 7.4, 1.7)
            visibility = st.slider("Visibility (10 m)", 27, 2000, 1434)
            dewpoint = st.slider("Dew point temperature (°C)", -30.0, 27.0, 7.0)
            solar = st.slider("Solar radiation (MJ/m²)", 0.0, 3.52, 0.5)
            rain = st.slider("Rainfall (mm)", 0.0, 35.0, 0.0)
            snow = st.slider("Snowfall (cm)", 0.0, 8.8, 0.0)
        submitted = st.form_submit_button("Generate forecasts", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "Count": 0,
            "Hour": hour,
            "Temp": temp,
            "Humidity": humidity,
            "Wind": wind,
            "Visibility": visibility,
            "DewPoint": dewpoint,
            "Solar": solar,
            "Rain": rain,
            "Snow": snow,
            "Season": season,
            "Holiday": int(holiday),
            "Month": month,
            "Weekend": int(weekend),
            "Previous hour count": previous_hour,
            "Previous day count": previous_day,
        }])
        encoded = encode_features(row).drop(columns=["Count"])
        # Ensure the input columns match the training matrix exactly.
        encoded = encoded.reindex(columns=X_train.columns, fill_value=0)
        pred_lr = float(model_a.predict(encoded)[0])
        pred_rf = float(model_b.predict(encoded)[0])
        p1, p2, p3 = st.columns(3)
        p1.metric("Linear Regression", f"{pred_lr:,.0f} bikes")
        p2.metric("Random Forest", f"{pred_rf:,.0f} bikes")
        p3.metric("Difference", f"{abs(pred_lr - pred_rf):,.0f} bikes")
        if pred_lr < 0 or pred_rf < 0:
            st.warning(
                "A regression model can produce a negative raw prediction even though bike counts cannot be negative. "
                "The displayed values are raw model outputs and have not been clipped."
            )
        st.caption(
            "These are model estimates for the entered conditions, not guaranteed real-world demand. "
            "Historical lag inputs are required because the forecasting models use previous-hour "
            "and previous-day demand."
        )


st.sidebar.caption("Built with Streamlit · Seoul Bike Sharing Demand")
