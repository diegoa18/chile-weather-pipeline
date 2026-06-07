import pandas as pd

FEATURE_LAGS = {
    "temp_avg": [1, 3, 7],
    "precipitation": [1, 3, 7],
    "humidity_avg": [1, 3, 7],
}

ROLLING_WINDOWS = [3, 7]
ROLLING_COLS = ["temp_avg", "precipitation"]


def add_temporal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = df.copy()

    if "date" not in df.columns:
        raise ValueError("columna 'date' no encontrada en el DataFrame")

    df = df.sort_values(by="date")

    created_features = []

    for col, lags in FEATURE_LAGS.items():
        if col not in df.columns:
            continue

        for lag in lags:
            feature_name = f"{col}_lag_{lag}"
            df[feature_name] = df[col].shift(lag)
            created_features.append(feature_name)

    for col in ROLLING_COLS:
        if col not in df.columns:
            continue

        for window in ROLLING_WINDOWS:
            feature_name = f"{col}_rolling_{window}"
            df[feature_name] = df[col].rolling(window).mean()
            created_features.append(feature_name)

    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    created_features.extend(["day_of_year", "month"])

    return df, created_features
