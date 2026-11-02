"""Step 1 - Data preparation: load, clean, create features, split into train/test."""
import pandas as pd
import config

SOURCE_COLUMNS = ["Date", "Rented Bike Count", "Hour", "Temperature(°C)",
                  "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)",
                  "Dew point temperature(°C)", "Solar Radiation (MJ/m2)",
                  "Rainfall(mm)", "Snowfall (cm)", "Seasons", "Holiday",
                  "Functioning Day"]


def load_and_clean(path=config.DATA_PATH):
    """Read, validate, clean, and chronologically order the source data."""
    df = pd.read_csv(path, encoding="latin1")          # file is not UTF-8
    if list(df.columns) != SOURCE_COLUMNS:
        raise ValueError(
            "Unexpected Seoul Bike CSV schema. Expected columns: "
            + ", ".join(SOURCE_COLUMNS)
        )
    df.columns = ["Date", "Count", "Hour", "Temp", "Humidity", "Wind",
                  "Visibility", "DewPoint", "Solar", "Rain", "Snow",
                  "Season", "Holiday", "Functioning"]
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("The CSV contains invalid dates in the Date column.")
    if df.isna().any().any():
        raise ValueError("The CSV contains missing values before cleaning.")
    if not set(df["Season"]).issubset({"Spring", "Summer", "Autumn", "Winter"}):
        raise ValueError("Season contains an unknown category.")
    if not set(df["Holiday"]).issubset({"Holiday", "No Holiday"}):
        raise ValueError("Holiday contains an unknown category.")
    if not set(df["Functioning"]).issubset({"Yes", "No"}):
        raise ValueError("Functioning contains an unknown category.")
    if not df["Hour"].between(0, 23).all():
        raise ValueError("Hour must contain values from 0 through 23.")

    report = {"rows_raw": len(df),
              "missing_values": int(df.isna().sum().sum()),
              "duplicate_rows": int(df.duplicated().sum())}

    # On non-functioning days no bikes can be rented (Count is always 0) -> drop them
    report["non_functioning_dropped"] = int((df["Functioning"] == "No").sum())
    df = df[df["Functioning"] == "Yes"].drop(columns="Functioning").copy()

    # Date-based features
    df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
    d = df["Date"]
    df["Month"] = d.dt.month
    df["Weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["Holiday"] = (df["Holiday"] == "Holiday").astype(int)
    # These are genuine historical predictors, available before the current hour.
    df["Previous hour count"] = df["Count"].shift(1)
    df["Previous day count"] = df["Count"].shift(24)
    if df.iloc[24:].isna().any().any():
        raise ValueError("Missing values remain after the lag warm-up period.")
    report["rows_clean"] = len(df)
    return df, report


def encode_features(df):
    """Turn the Season text column into 0/1 columns so models can use it."""
    encoded = pd.get_dummies(df.drop(columns=["Date"], errors="ignore"),
                              columns=["Season"], drop_first=True, dtype=int)
    return encoded


def split_data(df):
    """
    Chronological 70/30 split. Earlier observations are used for training and
    the latest observations are held out as a future-period test set.
    """
    ordered = (df.sort_values(["Date", "Hour"])
               .dropna(subset=["Previous hour count", "Previous day count"])
               .reset_index(drop=True))
    split_at = int(len(ordered) * (1 - config.TEST_SIZE))
    encoded = encode_features(ordered)
    train = encoded.iloc[:split_at]
    test = encoded.iloc[split_at:]
    y_train = train["Count"].values.astype(float)
    y_test = test["Count"].values.astype(float)
    X_train = train.drop(columns="Count")      # Count must NOT be an input feature
    X_test = test.drop(columns="Count")
    return X_train, X_test, y_train, y_test
