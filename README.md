# Seoul Bike Sharing Demand Analysis

## Abstract

This project analyses hourly bike rental demand in Seoul using descriptive
statistics, probability, inferential statistics, and supervised learning. The
main modelling objective is to compare a transparent Linear Regression model
with a nonlinear Random Forest model under a time-aware evaluation design.

Because the observations are ordered in time, the project does not use a
random row split. Earlier observations are used for training and later
observations are reserved for evaluation. This provides a more realistic
estimate of forecasting performance than mixing neighbouring hours between the
training and test sets.

For the operating-day dataset used here, Random Forest achieved lower error
than Linear Regression on the final chronological holdout and across the mean
of five time-series validation folds. This finding applies to the available
data and features; it is not intended as a universal claim about all bike
demand forecasting problems.

## Research objectives

The analysis addresses the following questions:

1. How is hourly bike demand distributed across time, seasons, and weather
   conditions?
2. What probability relationships can be observed between high demand, rain,
   rush hours, and season?
3. Does Random Forest improve forecasting accuracy compared with Linear
   Regression?
4. Does the comparison remain similar when evaluated using chronological
   holdout data and time-series cross-validation?
5. How should the statistical results be interpreted when hourly observations
   are temporally dependent?

## Dataset

The project uses the Seoul Bike Sharing Demand dataset stored at
`data/SeoulBikeData.csv`.

The source file contains 8,760 hourly records covering one year. Each record
contains:

- date and hour of observation;
- rented bike count;
- temperature, humidity, wind speed, visibility, dew-point temperature,
  solar radiation, rainfall, and snowfall;
- season;
- holiday status;
- whether the service was operating.

The modelling task is defined as forecasting demand on days when the service is
operating. The 295 records from non-functioning days are therefore excluded,
rather than being converted into zero-demand observations.

## Data preparation

The preprocessing pipeline performs the following steps:

1. Reads the CSV using the source file's `latin1` encoding.
2. Validates the expected 14-column schema.
3. Checks dates, missing values, category values, and hour ranges.
4. Removes records where the service was not functioning.
5. Sorts the remaining observations by date and hour.
6. Converts holiday status into a binary feature.
7. Creates `Month` and `Weekend` features.
8. Creates two historical demand features:
   - `Previous hour count`: the preceding available observation;
   - `Previous day count`: the value from 24 observations earlier.
9. Uses only rows with complete lag values for modelling.

The cleaned analysis dataset contains **8,465 hourly records**. The first 24
records do not have a complete 24-observation lag, so the modelling dataset
contains **8,441 records**. The 8,465-row dataset is retained for exploratory,
descriptive, probability, and inferential analysis.

The current hour's rental count is never included as a predictor. Historical
rental counts enter the models only through the lag variables.

## Exploratory and statistical analysis

The batch analysis produces:

- demand distributions and boxplots;
- average demand by hour;
- hourly demand patterns by season;
- seasonal comparisons;
- temperature-demand scatterplots;
- a numeric correlation matrix;
- descriptive summaries including mean, median, variance, quartiles,
  skewness, kurtosis, coefficient of variation, and IQR outliers;
- conditional probabilities involving high demand, rain, rush hours, and
  season;
- a check of independence between high demand and rainfall;
- the law of total probability and Bayes' theorem examples.

The inferential section includes:

- a confidence interval for mean hourly rentals;
- Welch's t-test for rainy versus dry hours;
- one-way ANOVA across seasons;
- a chi-square test for high demand and rainfall;
- Pearson correlation between temperature and demand;
- paired model-error comparisons using a t-test and Wilcoxon signed-rank test;
- bootstrap intervals for model RMSE and RMSE improvement;
- Central Limit Theorem demonstrations.

The conventional hypothesis tests are treated as exploratory. Hourly
observations within the same day are related, so their p-values should not be
interpreted as if the data were a collection of independent random samples.

## Forecasting methodology

### Forecasting approaches

Three approaches are evaluated:

1. **Previous-hour baseline**
   A rule-based benchmark that uses the most recently observed demand. It is
   not trained and provides a simple reference point.

2. **Linear Regression**
   A scaled, trained regression model using weather, calendar, seasonal, and
   historical-demand features.

3. **Random Forest**
   A 300-tree nonlinear regression model using the same feature set as Linear
   Regression.

The models are evaluated using:

- **RMSE**, which gives greater weight to large errors;
- **MAE**, the average absolute error in bikes per hour;
- **R²**, the proportion of target variation explained relative to predicting
  the test-set mean.

### Time-aware evaluation

The observations are ordered chronologically before modelling:

- the earliest 70% of observations are used for training;
- the latest 30% are used as a final chronological holdout.

The project also uses five expanding-window `TimeSeriesSplit` folds. In each
fold, the models are trained on earlier observations and evaluated on the next
time segment. Fold-level metrics, the mean RMSE, and the sample standard
deviation are saved in `outputs/results.json`.

### Bootstrap design

Hourly observations are not independent. For the official batch comparison,
the test-set errors are resampled using contiguous 24-observation blocks
rather than individual rows. This preserves some of the dependence between
nearby observations. The dashboard uses the same block structure for its
interactive bootstrap, although the number of resamples can be selected by
the user.

## Results

The checked-in results were generated using `SEED = 42`.

### Final chronological holdout

| Forecasting approach | RMSE | MAE | R² |
|---|---:|---:|---:|
| Previous-hour baseline | 325.96 | 227.04 | 0.7177 |
| Linear Regression | 275.36 | 187.95 | 0.7985 |
| Random Forest | **176.59** | **104.27** | **0.9171** |

Random Forest reduced RMSE by **98.77 bikes per hour** compared with Linear
Regression on the final holdout. The 95% block-bootstrap interval for this
improvement was **[83.76, 113.95] bikes per hour**.

### Five-fold time-series validation

| Model | Mean RMSE | Standard deviation |
|---|---:|---:|
| Linear Regression | 262.82 | 75.54 |
| Random Forest | **216.11** | 66.72 |

Random Forest had a lower mean RMSE across the five validation folds and a
lower RMSE on the final chronological holdout. The fold results are available
in `outputs/results.json`.

## Dashboard

The Streamlit dashboard provides five sections:

1. **Project Overview**
   Dataset counts, headline model metrics, validation information, and
   interpretation notes.

2. **Exploratory Data Analysis**
   Filters for season and hour, demand distributions, hourly patterns,
   seasonal summaries, weather plots, correlations, the full batch-generated
   figure set, and overall descriptive summaries.

3. **Model Comparison**
   RMSE, MAE, and R² tables, fold-level time-series validation, validation
   plots, official bootstrap results, feature importance, actual-versus-
   predicted plots, and error distributions.

4. **Statistical Analysis**
   Model comparison tests, confidence intervals, probability calculations,
   the complete probability results, dataset inference, bootstrap analysis,
   and CLT simulations.

5. **Bike Rental Prediction**
   An input form for Linear Regression and Random Forest predictions. The form
   includes previous-hour and previous-day demand because those values are
   required by the trained models.

The dashboard recalculates its interactive model views. The overview also
loads the saved batch results from `outputs/results.json`.

## Project structure

| File or directory | Purpose |
|---|---|
| `main.py` | Runs the complete batch analysis |
| `app.py` | Streamlit dashboard |
| `config.py` | Shared paths, seed, split size, significance level, and bootstrap size |
| `preprocessing.py` | Schema validation, cleaning, feature engineering, and chronological split |
| `models.py` | Forecasting baseline, model training, metrics, and prediction plots |
| `inference.py` | Confidence intervals, hypothesis tests, bootstrap, and CLT simulations |
| `descriptive_stats.py` | Descriptive and grouped summaries |
| `probability.py` | Conditional probability, independence, total probability, and Bayes calculations |
| `eda.py` | Batch exploratory figures |
| `tests/` | Automated tests for loading, features, lags, splitting, and predictions |
| `data/SeoulBikeData.csv` | Input dataset |
| `outputs/results.json` | Latest structured results |
| `outputs/run_log.txt` | Latest batch console log |
| `outputs/figures/` | Generated figures |
| `requirements.txt` | Version-constrained Python dependencies |

## Installation

The project was tested with Python 3.11. Python 3.9 or newer is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the project

Run the complete batch workflow:

```bash
python main.py
```

This regenerates:

- `outputs/results.json`;
- the EDA figures;
- the model prediction figure;
- the bootstrap and sampling figures;
- the CLT figures.

Run the tests:

```bash
pytest -q
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Reproducibility

The shared settings are defined in `config.py`:

- random seed: `42`;
- test proportion: `0.30`;
- significance level: `0.05`;
- official bootstrap resamples: `10,000`.

The dependency ranges are listed in `requirements.txt`. For reproducible
results, use the same Python version and regenerate the results from the
original CSV rather than editing `outputs/results.json` manually.

## Limitations

- The dataset covers one year and may not represent other years, locations, or
  operating conditions.
- Hourly observations are temporally dependent; individual rows are not
  independent samples.
- The previous-day feature is a 24-observation lag, not a calendar-aware
  same-hour lookup.
- Non-functioning days are excluded from the modelling task.
- The Random Forest configuration is a reasonable demonstration model, not an
  exhaustive hyperparameter search.
- The feature set does not include rolling averages or more advanced
  calendar-aware lag engineering.
- The inferential tests describe associations and should not be interpreted as
  causal evidence.
- The dashboard prediction form requires historical lag values to be supplied
  by the user.

## Conclusion

For the operating-day observations in this dataset, Random Forest performed
better than Linear Regression on the final chronological holdout and across
the average of five time-series validation folds. The result supports Random
Forest as the stronger model in this experiment, while the time-dependent
nature of the data means that the conclusion should be interpreted as
dataset-specific rather than universal.
