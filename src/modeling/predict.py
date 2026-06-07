import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.modeling.features.temporal_features import (
    FEATURE_LAGS,
    ROLLING_COLS,
    ROLLING_WINDOWS,
)
from src.utils.paths import city_slug, get_city_path
from src.utils.serializer import NumpyEncoder

logger = logging.getLogger(__name__)

TARGET = "temp_avg"


def load_model(city_name: str):
    folder = get_city_path(city_name, "models")
    path = folder / f"{city_slug(city_name)}_temp_model.pkl"

    if not path.exists():
        raise FileNotFoundError(f"no se encontro modelo para {city_name} en {path}")

    return joblib.load(path)


def _build_feature_vector(
    history: pd.DataFrame,
    feature_names: list[str],
    date: pd.Timestamp,
) -> pd.Series:
    vector = {}

    for feat in feature_names:
        if feat == "day_of_year":
            vector[feat] = date.dayofyear
            continue
        if feat == "month":
            vector[feat] = date.month
            continue

        # lags
        is_lag = False
        for col, lags in FEATURE_LAGS.items():
            for lag in lags:
                if feat == f"{col}_lag_{lag}":
                    if len(history) >= lag:
                        vector[feat] = history[col].iloc[-lag]
                    else:
                        vector[feat] = history[col].mean()
                    is_lag = True
                    break
            if is_lag:
                break
        if is_lag:
            continue

        # rolling
        is_rolling = False
        for col in ROLLING_COLS:
            for window in ROLLING_WINDOWS:
                if feat == f"{col}_rolling_{window}":
                    if len(history) >= window:
                        vector[feat] = history[col].tail(window).mean()
                    else:
                        vector[feat] = history[col].mean()
                    is_rolling = True
                    break
            if is_rolling:
                break
        if is_rolling:
            continue

        # features base: usar ultimo valor conocido
        if feat in history.columns:
            vector[feat] = history[feat].iloc[-1]
        else:
            vector[feat] = 0.0

    return pd.Series(vector)


def forecast_future(
    model_payload: dict,
    recent_data: pd.DataFrame,
    days_ahead: int = 3,
) -> pd.DataFrame:
    """
    forecast recursivo be like:

    1. Toma la ventana más reciente de datos historicos
    2. Predice temp_avg para t+1
    3. Agrega la prediccion al historial
    4. Reconstruye lags/rolling para t+2
    5. Repite
    """
    model = model_payload["model"]
    features = model_payload["features"]

    required_base = [
        c
        for c in features
        if c
        not in (
            [f"{col}_lag_{lag}" for col, lags in FEATURE_LAGS.items() for lag in lags]
            + [f"{col}_rolling_{w}" for col in ROLLING_COLS for w in ROLLING_WINDOWS]
            + ["day_of_year", "month"]
        )
    ]

    missing = [c for c in required_base if c not in recent_data.columns]
    if missing:
        raise ValueError(f"features base faltantes en recent_data: {missing}")

    df = recent_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    last_date = df["date"].max()

    predictions = []

    for step in range(1, days_ahead + 1):
        future_date = last_date + pd.Timedelta(days=step)

        x_vec = _build_feature_vector(df, features, future_date)
        pred = model.predict(pd.DataFrame([x_vec]))[0]

        predictions.append(
            {"date": future_date, "predicted_temp_avg": round(float(pred), 2)}
        )

        new_row = {
            "date": future_date,
            TARGET: pred,
        }
        for col in required_base:
            new_row[col] = df[col].iloc[-1]
        if "temp_max" in df.columns:
            new_row["temp_max"] = df["temp_max"].iloc[-1]
        if "temp_min" in df.columns:
            new_row["temp_min"] = df["temp_min"].iloc[-1]

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(predictions)


def save_forecast(city_name: str, forecast_df: pd.DataFrame, metrics: dict) -> str:
    folder = get_city_path(city_name, "results")

    slug = city_slug(city_name)
    csv_path = folder / f"{slug}_forecast.csv"
    forecast_df.to_csv(csv_path, index=False)

    json_path = folder / f"{slug}_model_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)

    logger.info("pronostico guardado en: %s", csv_path)
    return str(csv_path)
