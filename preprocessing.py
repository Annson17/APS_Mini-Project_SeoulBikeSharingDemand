"""Step 1 - Data preparation: load, clean, create features, split into train/test."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config


def load_and_clean(path=config.DATA_PATH):
    """Read the CSV and return a clean DataFrame (readable column names, no useless rows)."""
    df = pd.read_csv(path, encoding="latin1")          # file is not UTF-8
    df.columns = ["Date", "Count", "Hour", "Temp", "Humidity", "Wind",
                  "Visibility", "DewPoint", "Solar", "Rain", "Snow",
                  "Season", "Holiday", "Functioning"]

    report = {"rows_raw": len(df),
              "missing_values": int(df.isna().sum().sum()),
              "duplicate_rows": int(df.duplicated().sum())}

    # On non-functioning days no bikes can be rented (Count is always 0) -> drop them
    report["non_functioning_dropped"] = int((df["Functioning"] == "No").sum())
    df = df[df["Functioning"] == "Yes"].drop(columns="Functioning").copy()

    # Date-based features
    d = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["Month"] = d.dt.month
    df["Weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["Holiday"] = (df["Holiday"] == "Holiday").astype(int)
    df = df.drop(columns="Date").reset_index(drop=True)

    report["rows_clean"] = len(df)
    return df, report


def encode_features(df):
    """Turn the Season text column into 0/1 columns so models can use it."""
    return pd.get_dummies(df, columns=["Season"], drop_first=True, dtype=int)


def split_data(df):
    """
    Random 70/30 split. The TARGET is the numeric column 'Count'
    (bikes rented in that hour), so this is a regression problem.
    """
    df_enc = encode_features(df)
    train, test = train_test_split(df_enc, test_size=config.TEST_SIZE,
                                   random_state=config.SEED)
    y_train = train["Count"].values.astype(float)
    y_test = test["Count"].values.astype(float)
    X_train = train.drop(columns="Count")      # Count must NOT be an input feature
    X_test = test.drop(columns="Count")
    return X_train, X_test, y_train, y_test
