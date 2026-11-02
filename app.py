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
from models import train_models

st.set_page_config(
    page_title="Seoul Bike Analytics",
    page_icon="🚲",
    layout="wide",
)

sns.set_theme(style="whitegrid")
ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "outputs" / "results.json"


@st.cache_data(show_spinner="Loading and cleaning Seoul Bike data...")
def get_data():
    df, report = load_and_clean()
    return df, report


@st.cache_data(show_spinner="Preparing the train/test split...")
def get_split(_df):
    # Use the same fixed random split and feature engineering as the original project.
    return split_data(_df)


@st.cache_resource(show_spinner="Training the models for the first time...")
def get_models(_X_train, _y_train):
    # Model objects are cached for this Streamlit session/server process.
    return train_models(_X_train, _y_train)


@st.cache_data(show_spinner="Evaluating models...")
def get_predictions(_model_a, _model_b, X_test, y_test):
    pred_a = _model_a.predict(X_test)
    pred_b = _model_b.predict(X_test)
    metrics = {
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


def make_fig():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    return fig, ax


def show_metric_table(metrics):
    st.dataframe(pd.DataFrame(metrics).T.style.format({
        "RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}"
    }), use_container_width=True)


try:
    df, prep_report = get_data()
    X_train, X_test, y_train, y_test = get_split(df)
    model_a, model_b = get_models(X_train, y_train)
    pred_a, pred_b, metrics = get_predictions(model_a, model_b, X_test, y_test)
except Exception as exc:
    st.error("The project could not load its data or models.")
    st.exception(exc)
    st.info(
        "Check that app.py is in the APS_Project folder, the dataset is at "
        "data/SeoulBikeData.csv, and the original Python modules are beside app.py."
    )
    st.stop()


with st.sidebar:
    st.title("🚲 Seoul Bike Analytics")
    st.caption("Applied Probability & Statistics")
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
    st.caption(f"Cleaned dataset: {len(df):,} hourly records")
    st.caption("Split: 70% training / 30% testing")


# ---------------------------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------------------------
if page == "🏠 Project Overview":
    st.title("🚲 Is the Improved Model Really Better?")
    st.subheader("Interactive Seoul Bike Sharing Demand Analysis")
    st.write(
        "This dashboard combines exploratory analysis, probability, inferential "
        "statistics and a comparison of two regression models predicting hourly bike rentals."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw records", f"{prep_report['rows_raw']:,}")
    c2.metric("Clean records", f"{prep_report['rows_clean']:,}")
    c3.metric("Training records", f"{len(X_train):,}")
    c4.metric("Test records", f"{len(X_test):,}")

    st.markdown("### Main findings")
    m1, m2, m3 = st.columns(3)
    m1.metric("Linear Regression RMSE", f"{metrics['Linear Regression']['RMSE']:.1f}")
    m2.metric("Random Forest RMSE", f"{metrics['Random Forest']['RMSE']:.1f}")
    improvement = metrics["Linear Regression"]["RMSE"] - metrics["Random Forest"]["RMSE"]
    m3.metric("RMSE reduction", f"{improvement:.1f} bikes/hour")

    st.markdown("### Project workflow")
    st.markdown(
        """
        1. **Prepare the data:** remove non-functioning-day rows and derive month/weekend features.
        2. **Explore the data:** inspect demand distributions, hourly patterns, seasons and weather.
        3. **Study probability:** conditional probability, independence, total probability and Bayes' theorem.
        4. **Train two models:** Linear Regression as the baseline and Random Forest as the improved model.
        5. **Evaluate uncertainty:** paired t-test, bootstrap confidence intervals, Wilcoxon test and CLT demonstrations.
        """
    )
    st.warning(
        "Interpretation caveat: the original project uses a random hourly split. "
        "Neighbouring hours can be correlated, so independence assumptions in inferential tests "
        "and row-based bootstrap intervals may be optimistic."
    )
    with st.expander("Data preparation details"):
        st.json(prep_report)


# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.write("Use the filters to explore how hourly rentals vary across time, season and weather.")

    seasons = sorted(df["Season"].dropna().unique().tolist())
    chosen_seasons = st.multiselect("Select seasons", seasons, default=seasons)
    hour_range = st.slider("Hour range", 0, 23, (0, 23))
    filtered = df[
        df["Season"].isin(chosen_seasons)
        & df["Hour"].between(hour_range[0], hour_range[1])
    ].copy()

    if filtered.empty:
        st.warning("No records match these filters.")
        st.stop()

    a, b, c = st.columns(3)
    a.metric("Filtered hours", f"{len(filtered):,}")
    b.metric("Mean rentals", f"{filtered['Count'].mean():.1f}")
    c.metric("Median rentals", f"{filtered['Count'].median():.1f}")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Time & seasons", "Weather & correlation"])
    with tab1:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(filtered["Count"], bins=40, kde=True, ax=ax[0])
        ax[0].set_title("Hourly rental distribution")
        sns.boxplot(data=filtered, x="Count", ax=ax[1])
        ax[1].set_title("Hourly rental boxplot")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        hourly = filtered.groupby("Hour", as_index=False)["Count"].mean()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.lineplot(data=hourly, x="Hour", y="Count", marker="o", ax=ax)
        ax.set_title("Mean rentals by hour")
        ax.set_xticks(range(hour_range[0], hour_range[1] + 1))
        ax.set_ylabel("Mean rentals")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        seasonal = filtered.groupby("Season", as_index=False)["Count"].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=seasonal, x="Season", y="Count", ax=ax)
        ax.set_title("Mean rentals by season")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(data=filtered.sample(min(1800, len(filtered)), random_state=42),
                        x="Temp", y="Count", hue="Season", alpha=0.55, ax=ax)
        ax.set_title("Temperature vs rentals")
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


# ---------------------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------------------
elif page == "🧠 Model Comparison":
    st.title("🧠 Model Comparison")
    st.write("Both models are evaluated on the same held-out test observations.")

    show_metric_table(metrics)
    st.markdown("### Compare metrics")
    metric_choice = st.selectbox("Metric", ["RMSE", "MAE", "R²"])
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(metrics.keys())
    values = [metrics[n][metric_choice] for n in names]
    ax.bar(names, values)
    ax.set_ylabel(metric_choice)
    ax.set_title(f"{metric_choice}: lower is better" if metric_choice != "R²" else "R²: higher is better")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Actual vs predicted")
    n_points = st.slider("Number of test observations to display", 100, len(y_test), min(800, len(y_test)), step=100)
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
        ax.set_xlabel("Actual rentals")
        ax.set_ylabel("Predicted rentals")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Error distributions")
    errors = pd.DataFrame({
        "Linear Regression": np.asarray(y_test) - pred_a,
        "Random Forest": np.asarray(y_test) - pred_b,
    })
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(errors["Linear Regression"], color="#4C72B0", stat="density",
                 element="step", fill=False, label="Linear Regression", ax=ax)
    sns.histplot(errors["Random Forest"], color="#55A868", stat="density",
                 element="step", fill=False, label="Random Forest", ax=ax)
    ax.set_title("Prediction error (actual − predicted)")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.info(
        "RMSE penalizes large errors more strongly; MAE is the average absolute error; "
        "R² describes the fraction of target variation explained relative to predicting the test mean."
    )


# ---------------------------------------------------------------------
# STATISTICAL ANALYSIS
# ---------------------------------------------------------------------
elif page == "🧪 Statistical Analysis":
    st.title("🧪 Statistical Analysis")
    st.write("Explore the statistical evidence behind the project conclusions.")
    tab1, tab2, tab3 = st.tabs(["Model comparison", "Data inference", "Probability & CLT"])

    with tab1:
        st.subheader("Paired t-test on squared errors")
        st.latex(r"d_i=(y_i-\hat y_{LR,i})^2-(y_i-\hat y_{RF,i})^2")
        st.write("H₀: the mean paired squared-error difference is zero.")
        st.write("H₁: the mean paired squared-error difference is not zero.")
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
        st.caption("Assumptions caveat: the paired t-test treats per-observation differences as independent; hourly bike data may have temporal dependence.")

        st.subheader("Paired bootstrap confidence interval for RMSE improvement")
        reps = st.slider("Bootstrap resamples", 500, 5000, 2000, step=500)
        rng = np.random.default_rng(42)
        n = len(y_test)
        improvements = np.empty(reps)
        # Chunking keeps memory use modest.
        chunk_size = 200
        for start in range(0, reps, chunk_size):
            stop = min(start + chunk_size, reps)
            indices = rng.integers(0, n, size=(stop - start, n))
            rmse_lr = np.sqrt(se_a[indices].mean(axis=1))
            rmse_rf = np.sqrt(se_b[indices].mean(axis=1))
            improvements[start:stop] = rmse_lr - rmse_rf
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
            st.success("The bootstrap interval is entirely above zero, supporting lower RMSE for Random Forest under this resampling procedure.")

        with st.expander("Wilcoxon signed-rank test (absolute errors)"):
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                np.abs(np.asarray(y_test) - pred_a),
                np.abs(np.asarray(y_test) - pred_b),
            )
            st.write(f"Statistic: **{wilcoxon_stat:,.1f}**")
            st.write(f"p-value: **{wilcoxon_p:.3e}**")
            st.caption("This is a paired, non-parametric comparison of absolute errors; it still requires careful interpretation when observations are dependent.")

    with tab2:
        st.subheader("Confidence interval for mean rentals")
        c = df["Count"].astype(float)
        alpha_data = 0.05
        lo, hi = stats.t.interval(0.95, df=len(c)-1, loc=c.mean(), scale=stats.sem(c))
        st.write(f"Mean hourly rentals: **{c.mean():.2f}**")
        st.write(f"95% t confidence interval: **[{lo:.2f}, {hi:.2f}]**")

        st.subheader("Welch's t-test: rainy vs dry hours")
        rain = df.loc[df["Rain"] > 0, "Count"]
        dry = df.loc[df["Rain"] == 0, "Count"]
        t_rain, p_rain = stats.ttest_ind(rain, dry, equal_var=False)
        r1, r2, r3 = st.columns(3)
        r1.metric("Mean on rainy hours", f"{rain.mean():.1f}")
        r2.metric("Mean on dry hours", f"{dry.mean():.1f}")
        r3.metric("p-value", f"{p_rain:.3e}")
        st.caption(f"Welch t-statistic = {t_rain:.2f}; rainy n = {len(rain):,}, dry n = {len(dry):,}.")

        st.subheader("One-way ANOVA: rentals across seasons")
        groups = [g["Count"].to_numpy() for _, g in df.groupby("Season")]
        f_stat, p_anova = stats.f_oneway(*groups)
        st.write(f"F-statistic: **{f_stat:.2f}** · p-value: **{p_anova:.3e}**")
        st.caption("A significant ANOVA indicates that at least one seasonal mean differs; it does not by itself identify which pairs differ.")

        st.subheader("Chi-square test: high demand and rain")
        high = df["Count"] > df["Count"].median()
        table = pd.crosstab(high, df["Rain"] > 0)
        chi2, p_chi, dof, expected = stats.chi2_contingency(table)
        st.write(f"χ² = **{chi2:.2f}**, degrees of freedom = **{dof}**, p-value = **{p_chi:.3e}**")
        st.dataframe(table.rename(index={False: "Not high", True: "High"},
                                  columns={False: "No rain", True: "Rain"}))

        st.subheader("Pearson correlation: temperature vs rentals")
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
        st.caption("High demand is defined here as rentals above the cleaned dataset median; rain means rainfall > 0 mm; rush hours are 7–9 and 17–19.")

        st.subheader("Central Limit Theorem demonstration")
        values_source = st.selectbox("Population to sample from", ["Hourly rentals", "Squared-error differences"])
        values = df["Count"].to_numpy(dtype=float) if values_source == "Hourly rentals" else d.astype(float)
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
        st.caption("As sample size grows, the distribution of sample means generally becomes less skewed and its standard error shrinks. This simulation does not remove the real dataset's temporal-dependence limitation.")


# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------
elif page == "🔮 Bike Rental Prediction":
    st.title("🔮 Predict Hourly Bike Rentals")
    st.write("Change the inputs and compare both trained regression models.")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            hour = st.slider("Hour of day", 0, 23, 18)
            month = st.slider("Month", 1, 12, 7)
            weekend = st.selectbox("Day type", ["Weekday", "Weekend"]) == "Weekend"
            season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
            holiday = st.selectbox("Holiday status", ["No holiday", "Holiday"]) == "Holiday"
            temp = st.slider("Temperature (°C)", -17.8, 39.4, 15.0)
            humidity = st.slider("Humidity (%)", 0, 100, 58)
        with right:
            wind = st.slider("Wind speed (m/s)", 0.0, 7.4, 1.7)
            visibility = st.slider("Visibility (10 m)", 27, 2000, 1434)
            dewpoint = st.slider("Dew point temperature (°C)", -30.0, 27.0, 7.0)
            solar = st.slider("Solar radiation (MJ/m²)", 0.0, 3.52, 0.5)
            rain = st.slider("Rainfall (mm)", 0.0, 35.0, 0.0)
            snow = st.slider("Snowfall (cm)", 0.0, 8.8, 0.0)
        submitted = st.form_submit_button("Predict rentals", type="primary")

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
            "The training data's feature ranges and the random train/test methodology limit generalization."
        )


st.sidebar.caption("Built with Streamlit · Seoul Bike Sharing Demand")
