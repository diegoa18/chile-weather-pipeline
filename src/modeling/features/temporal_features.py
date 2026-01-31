import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = df.copy()

    if "date" not in df.columns:
        raise ValueError("columna 'date' no encontrada en el DataFrame")

    df = df.sort_values(by="date")

    created_features = []

    lag_features = {
        "temp_avg": [1, 3, 7],
        "precipitation": [1, 3, 7],
        "humidity_avg": [1, 3, 7],
    }

    for col, lags in lag_features.items():
        if col not in df.columns:
            continue

        for lag in lags:
            feature_name = f"{col}_lag_{lag}"
            df[feature_name] = df[col].shift(lag)
            created_features.append(feature_name)

    rolling_windows = [3, 7]
    rolling_cols = ["temp_avg", "precipitation"]

    for col in rolling_cols:
        if col not in df.columns:
            continue

        for window in rolling_windows:
            feature_name = f"{col}_rolling_{window}"
            df[feature_name] = df[col].rolling(window).mean()
            created_features.append(feature_name)

    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    created_features.extend(["day_of_year", "month"])

    return df, created_features
