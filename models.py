"""Step 5 - Model training and evaluation (regression: predict hourly bike count)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import config


def train_models(X_train, y_train):
    """Train the statistical baseline and the nonlinear comparison model."""
    baseline = make_pipeline(StandardScaler(), LinearRegression())
    improved = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)
    baseline.fit(X_train, y_train)
    improved.fit(X_train, y_train)
    return baseline, improved


def lag_baseline(X):
    """Return the previous-hour demand baseline, available from historical data."""
    return np.asarray(X["Previous hour count"], dtype=float)


def evaluate(model, X_test, y_test, name):
    """RMSE (typical error in bikes/hour), MAE (average absolute error), R2 (share of variance explained)."""
    pred = model(X_test) if callable(model) else model.predict(X_test)
    m = {"rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
         "mae": float(mean_absolute_error(y_test, pred)),
         "r2": float(r2_score(y_test, pred))}
    print(f"{name:24s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    return pred, m


def plot_predictions(y_test, pred_a, pred_b):
    """Predicted vs actual rentals: points on the red line = perfect predictions."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for a, pred, t in [(ax[0], pred_a, "Baseline (Linear Regression)"),
                       (ax[1], pred_b, "Improved (Random Forest)")]:
        a.scatter(y_test, pred, s=6, alpha=0.4)
        lim = [0, max(y_test.max(), pred.max())]
        a.plot(lim, lim, color="red", lw=1)
        a.set_title(t); a.set_xlabel("Actual rentals")
    ax[0].set_ylabel("Predicted rentals")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig_pred_vs_actual.png"), dpi=180)
    plt.close()
