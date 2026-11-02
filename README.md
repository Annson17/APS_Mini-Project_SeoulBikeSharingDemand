# Does the improved ML model really beat the baseline? (Seoul Bike Sharing Demand)

Regression task: predict the number of bikes rented per hour.
Baseline = Linear Regression, Improved = Random Forest Regressor.
Probability and statistics (sampling, estimation, confidence intervals, hypothesis testing)
decide whether the improvement is real or just test-sample variation.

## Project structure
| File | Purpose |
|---|---|
| `config.py` | Paths, random seed, alpha, bootstrap size |
| `preprocessing.py` | Load, clean, feature creation, train/test split (target = hourly bike count) |
| `eda.py` | Exploratory plots (distribution, hourly pattern, season, correlation) |
| `descriptive_stats.py` | Mean, median, SD, quartiles, skewness, outliers, group summaries |
| `probability.py` | Conditional probability, independence, total probability, Bayes' theorem |
| `models.py` | Linear Regression and Random Forest training; RMSE, MAE, R2; predicted-vs-actual plot |
| `inference.py` | t-interval, Welch t-test, ANOVA, chi-square, Pearson; bootstrap CI for RMSE, paired t-test, Wilcoxon test, sampling demo |
| `main.py` | Runs everything in order |

## How to run
```bash
pip install -r requirements.txt
python main.py
```
`SeoulBikeData.csv` is in `data/`. Results go to `outputs/results.json` and `outputs/run_log.txt`; figures go to `outputs/figures/`.
Everything is seeded (`SEED = 42`), so results are reproducible.
