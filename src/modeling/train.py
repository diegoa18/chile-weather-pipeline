import json
import os
from asyncio.tasks import ensure_future
from tempfile import TemporaryFile
from tokenize import INDENT

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.config.settings import MODEL_PARAMS_RF, MODEL_PARAMS_XGB
from src.modeling.features import add_temporal_features
from src.utils.paths import get_city_path

# intento de importar XGBoost, fallback a Random Forest
try:
    from xgboost import XGBRegressor

    USE_XGB = True
except ImportError:
    USE_XGB = False


def prepare_training_data(df: pd.DataFrame):
    """prepara features (X) y target (y) para el entrenamiento."""
    if df.empty:
        raise ValueError("el DF para entrenamiento esta vacio")

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values("date")

    # seleccion de features potenciales
    base_features = [
        "precipitation",
        "windspeed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "humidity_avg",
        "dew_point_avg",
        "cloud_cover_mean",
        "solar_energy",
        "temp_range",
    ]

    # filtrar solo las columnas que existen en el DF
    available_base_features = [c for c in base_features if c in df.columns]

    if not available_base_features:
        raise ValueError("no hay features base disponibles para entrenar el modelo")

    # features temporales
    df, temporal_features = add_temporal_features(df)

    df = df.dropna().reset_index(drop=True)

    # matrice finales
    feature_columns = available_base_features + temporal_features

    x = df[feature_columns].fillna(0)
    y = df["temp_avg"]

    return x, y, feature_columns


def train_temperature_model(X, y):
    """entrenar el modelo (XGBoost o Random Forest)."""
    if USE_XGB:
        model = XGBRegressor(**MODEL_PARAMS_XGB)
    else:
        model = RandomForestRegressor(**MODEL_PARAMS_RF)

    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    """evalua el modelo usando Hold-Out set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model.fit(X_train, y_train)  # re-entrenamiento en train set para evaluación honesta
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}
    print(f"metricas del modelo: {metrics}")

    # re-entrenar con todo el dataset para producción
    model.fit(X, y)

    return metrics


def save_model(city_name: str, model, features: list) -> str:
    """guardar el modelo entrenado y los nombres de las features."""
    folder = get_city_path(city_name, "models")
    path = folder / f"{city_name.lower()}_temp_model.pkl"

    payload = {
        "model": model,
        "features": features,
        "model_type": "xgboost" if USE_XGB else "random_forest",
    }

    joblib.dump(payload, path)
    print(f"modelo guardado en: {path}")
    return str(path)


def save_feature_metadata(city_name: str, features: list) -> str:
    """guardar la metadata de las features"""
    folder = get_city_path(city_name, "results")
    path = folder / f"{city_name.lower()}_features.json"

    payload = {"total_features": len(features), "features": features}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    print(f"features guardadas en: {path}")
    return str(path)
