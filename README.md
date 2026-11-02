# Seoul Bike Sharing Demand Analysis

An Applied Probability and Statistics project that analyzes hourly bike rentals in Seoul and tests whether a Random Forest model materially outperforms a Linear Regression baseline.

The project has two ways to use it:

- **Batch analysis:** runs the complete data preparation, EDA, descriptive statistics, probability, inference, model comparison, and Central Limit Theorem demonstrations.
- **Interactive dashboard:** provides filters, charts, statistical tests, model comparisons, and an interactive rental prediction form in Streamlit.

## Research question

> Does the improved Random Forest model predict hourly bike rentals better than the Linear Regression baseline, and is the observed improvement likely to be more than test-sample variation?

## Results at a glance

The checked-in results were generated with `SEED = 42`, a 70/30 random train/test split, and the cleaned dataset:

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| RMSE (bikes/hour) | 423.94 | **171.30** |
| MAE (bikes/hour) | 323.95 | **99.03** |
| R² | 0.5518 | **0.9268** |

- The cleaned dataset contains **8,465 hourly records** after removing 295 non-functioning-day rows.
- The Random Forest reduces RMSE by **252.64 bikes/hour**.
- The paired bootstrap 95% confidence interval for the RMSE improvement is **[230.66, 273.56]** bikes/hour.
- A paired t-test on squared errors gives `p = 1.41e-96`; the Wilcoxon signed-rank check gives `p = 2.22e-302`.
- The Random Forest has the smaller squared error on **85.0%** of test hours.
- Average demand is **729.16 bikes/hour** (95% t-interval: **[715.47, 742.84]**).
- Mean demand is higher in summer (1,034.1) than winter (225.5), and rainy hours average 167.3 rentals versus 765.6 on dry hours.

These are associations and held-out test-set results, not causal claims.

## Dashboard preview

Run the Streamlit app to explore:

- Project overview and headline findings
- Filterable exploratory analysis by season and hour
- Model metrics, prediction scatterplots, and error distributions
- Confidence intervals, Welch's t-test, ANOVA, chi-square, Pearson correlation, bootstrap, paired t-test, and Wilcoxon tests
- Probability examples and Central Limit Theorem simulations
- A form for comparing Linear Regression and Random Forest rental predictions

## Project workflow

`main.py` executes the following stages:

1. **Preprocessing**
   - Reads `data/SeoulBikeData.csv` using `latin1` encoding.
   - Renames the source columns to concise names.
   - Reports missing and duplicate rows.
   - Removes rows where `Functioning Day` is `No`.
   - Derives `Month` and a binary `Weekend` feature.
   - Converts `Holiday` to a binary feature and removes the original date/functioning-day columns.
2. **Exploratory data analysis**
   - Creates distributions, hourly patterns, seasonal comparisons, temperature scatterplots, and a correlation heatmap.
3. **Descriptive statistics**
   - Calculates location, spread, quartiles, skewness, kurtosis, coefficient of variation, IQR outliers, seasonal summaries, and feature correlations.
4. **Probability**
   - Defines high demand as rentals above the cleaned-data median.
   - Examines rain, rush hours, conditional probability, independence, the law of total probability, and Bayes' theorem.
5. **Inferential statistics**
   - Computes a confidence interval for mean rentals.
   - Tests rainy versus dry demand with Welch's t-test.
   - Tests seasonal differences with one-way ANOVA.
   - Tests high demand/rain independence with chi-square.
   - Tests temperature/rental association with Pearson correlation.
6. **Model training**
   - Encodes `Season` with one-hot encoding.
   - Uses a reproducible 70/30 random split.
   - Trains a scaled Linear Regression baseline and a 300-tree Random Forest regressor.
7. **Model comparison**
   - Evaluates RMSE, MAE, and R² on the same held-out observations.
   - Uses paired bootstrap intervals, a paired t-test on squared errors, a Wilcoxon signed-rank test, and sampling-distribution simulations.
8. **CLT demonstrations**
   - Simulates sampling distributions for hourly rentals and per-hour squared-error differences at sample sizes 1, 5, 30, and 200.

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the batch analysis

From the project root:

```bash
python main.py
```

The run prints each analysis stage and writes:

- `outputs/results.json` - structured metrics, statistical results, and simulation summaries
- `outputs/run_log.txt` - example output from a completed run
- `outputs/figures/` - generated EDA, prediction, bootstrap, sampling, and CLT figures

Random operations use `SEED = 42` in `config.py`. Change the shared settings there rather than editing individual modules.

## Run the interactive dashboard

```bash
streamlit run app.py
```

The dashboard loads and cleans the data, trains both models, and caches the data, models, and predictions for the current Streamlit process. Open the local URL printed by Streamlit in a browser.

## Data

The input file is [`data/SeoulBikeData.csv`](data/SeoulBikeData.csv), containing hourly observations with:

| Source field | Meaning |
|---|---|
| `Date` | Observation date (`DD/MM/YYYY`) |
| `Rented Bike Count` | Hourly rental target |
| `Hour` | Hour of day, 0-23 |
| `Temperature(°C)` | Temperature |
| `Humidity(%)` | Relative humidity |
| `Wind speed (m/s)` | Wind speed |
| `Visibility (10m)` | Visibility |
| `Dew point temperature(°C)` | Dew-point temperature |
| `Solar Radiation (MJ/m2)` | Solar radiation |
| `Rainfall(mm)` | Rainfall |
| `Snowfall (cm)` | Snowfall |
| `Seasons` | Spring, Summer, Autumn, or Winter |
| `Holiday` | Holiday status |
| `Functioning Day` | Whether the bike service was operating |

The preprocessing module maps these to `Date`, `Count`, `Hour`, `Temp`, `Humidity`, `Wind`, `Visibility`, `DewPoint`, `Solar`, `Rain`, `Snow`, `Season`, `Holiday`, and `Functioning`.

## Repository structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `main.py` | Batch pipeline entry point |
| `config.py` | Paths, seed, split size, significance level, and bootstrap size |
| `preprocessing.py` | Data loading, cleaning, feature engineering, and train/test split |
| `eda.py` | Exploratory plots |
| `descriptive_stats.py` | Descriptive and grouped summaries |
| `probability.py` | Probability and Bayes calculations |
| `inference.py` | Confidence intervals, hypothesis tests, bootstrap, and CLT simulations |
| `models.py` | Model training, metrics, and prediction plot |
| `data/` | Input CSV |
| `outputs/results.json` | Latest structured batch results |
| `outputs/run_log.txt` | Latest recorded console output |
| `outputs/figures/` | Generated figures |
| `requirements.txt` | Python dependencies |
| `APS_source.zip` | Packaged source archive included with the project |

## Important limitations

- The split is random by hourly row, not time-based. Nearby hours can be correlated, so the test set may be easier than a future-period forecast.
- The paired tests and row bootstrap treat the test observations as sufficiently independent; temporal dependence can make uncertainty estimates optimistic.
- The analysis removes non-functioning days rather than modeling zero demand on those days.
- Random Forest predictions are raw regression outputs and are not clipped to zero.
- The results are observational: weather, season, time, and demand relationships should not be interpreted as causal without a stronger study design.
