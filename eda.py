"""Step 2 - Exploratory Data Analysis (EDA): pictures that reveal patterns in the data."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import config

sns.set_theme(style="whitegrid")


def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, name), dpi=180)
    plt.close()


def run_eda(df):
    # 1. Distribution of the target variable
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["Count"], bins=40, kde=True, ax=ax[0], color="#4C72B0")
    ax[0].set_title("Distribution of hourly bike rentals")
    sns.boxplot(x=df["Count"], ax=ax[1], color="#55A868")
    ax[1].set_title("Boxplot (dots = outliers)")
    _save("eda_count_distribution.png")

    # 2. Daily pattern: average rentals per hour of the day
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=df, x="Hour", y="Count", estimator="mean", errorbar=None, marker="o")
    plt.title("Average rentals by hour of day (rush-hour peaks)")
    _save("eda_hourly_pattern.png")

    # 3. Demand by season
    order = ["Spring", "Summer", "Autumn", "Winter"]
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Season", y="Count", order=order)
    plt.title("Rentals by season")
    _save("eda_season_boxplot.png")


    # 3b. Daily pattern by season: average rentals per hour, one line per season
    plt.figure(figsize=(9.5, 5))
    sns.lineplot(data=df, x="Hour", y="Count", hue="Season", hue_order=order,
                 estimator="mean", errorbar=None, marker="o",
                 palette={"Spring": "#2ca02c", "Summer": "#d62728",
                          "Autumn": "#ff7f0e", "Winter": "#1f77b4"})
    plt.xticks(range(24))
    plt.title("Average bike rentals by hour, for each season")
    plt.ylabel("Average rentals per hour"); plt.xlabel("Hour of day")
    _save("eda_hourly_by_season.png")

    # 4. Temperature vs rentals
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df.sample(2000, random_state=config.SEED),
                    x="Temp", y="Count", alpha=0.4)
    plt.title("Temperature vs rentals (2000 random hours)")
    plt.xlabel("Temperature (C)")
    _save("eda_temp_scatter.png")

    # 5. Correlation heatmap of numeric columns
    num = df[["Count", "Hour", "Temp", "Humidity", "Wind", "Visibility",
              "DewPoint", "Solar", "Rain", "Snow"]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    _save("eda_correlation.png")

    return {"figures": ["eda_count_distribution", "eda_hourly_pattern", "eda_season_boxplot", "eda_hourly_by_season",
                        "eda_temp_scatter", "eda_correlation"]}
