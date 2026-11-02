import numpy as np

from models import lag_baseline, train_models
from preprocessing import encode_features, load_and_clean, split_data


def test_load_and_clean_validates_expected_dataset():
    df, report = load_and_clean()
    assert report["rows_raw"] == 8760
    assert report["non_functioning_dropped"] == 295
    assert len(df) == 8465
    assert df["Date"].is_monotonic_increasing
    assert df["Count"].isna().sum() == 0
    assert df["Previous day count"].isna().sum() == 24


def test_split_is_chronological_and_has_no_target_leakage():
    df, _ = load_and_clean()
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) + len(X_test) == len(df) - 24
    assert "Count" not in X_train.columns
    assert "Count" not in X_test.columns
    assert "Previous hour count" in X_train.columns
    assert X_train.index.max() < X_test.index.min()
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_encoding_and_models_produce_finite_metrics():
    df, _ = load_and_clean()
    X_train, X_test, y_train, y_test = split_data(df)
    baseline, forest = train_models(X_train, y_train)
    predictions = [lag_baseline(X_test), baseline.predict(X_test), forest.predict(X_test)]
    assert all(np.isfinite(pred).all() for pred in predictions)
    assert set(encode_features(df).columns).issuperset(X_train.columns)


def test_lag_features_are_historical_values():
    df, _ = load_and_clean()
    assert (df["Previous hour count"].iloc[1:].to_numpy()
            == df["Count"].iloc[:-1].to_numpy()).all()
    assert (df["Previous day count"].iloc[24:].to_numpy()
            == df["Count"].iloc[:-24].to_numpy()).all()
