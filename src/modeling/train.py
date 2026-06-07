import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config.settings import MODEL_PARAMS_RF, MODEL_PARAMS_XGB
from src.modeling.features import add_temporal_features
from src.utils.paths import city_slug, get_city_path
from src.utils.serializer import NumpyEncoder

logger = logging.getLogger(__name__)

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

    base_features = [
        "precipitation",
        "windspeed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "humidity_avg",
        "dew_point_avg",
        "cloud_cover_mean",
        "temp_range",
    ]

    available_base_features = [c for c in base_features if c in df.columns]

    if not available_base_features:
        raise ValueError("no hay features base disponibles para entrenar el modelo")

    df, temporal_features = add_temporal_features(df)

    df = df.dropna().reset_index(drop=True)

    feature_columns = available_base_features + temporal_features

    x = df[feature_columns].fillna(0)
    y = df["temp_avg"]

    return x, y, feature_columns


def _make_model():
    if USE_XGB:
        return XGBRegressor(**MODEL_PARAMS_XGB)
    return RandomForestRegressor(**MODEL_PARAMS_RF)


def evaluate_model(X, y):
    tscv = TimeSeriesSplit(n_splits=3)

    mae_scores, rmse_scores = [], []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = _make_model()
        fold_model.fit(X_train, y_train)
        preds = fold_model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, preds))
        rmse_scores.append(mean_squared_error(y_test, preds) ** 0.5)
        all_y_true.extend(y_test.values)
        all_y_pred.extend(preds)

    metrics = {
        "MAE": round(np.mean(mae_scores), 2),
        "MAE_std": round(np.std(mae_scores), 2),
        "RMSE": round(np.mean(rmse_scores), 2),
        "RMSE_std": round(np.std(rmse_scores), 2),
    }
    logger.info("metricas del modelo (3-fold TSS): %s", metrics)

    return metrics, np.array(all_y_true), np.array(all_y_pred)


def train_temperature_model(X, y):
    model = _make_model()
    model.fit(X, y)
    return model


def save_model(city_name: str, model, features: list) -> str:
    """guardar el modelo entrenado y los nombres de las features."""
    folder = get_city_path(city_name, "models")
    path = folder / f"{city_slug(city_name)}_temp_model.pkl"

    payload = {
        "model": model,
        "features": features,
        "model_type": "xgboost" if USE_XGB else "random_forest",
    }

    joblib.dump(payload, path)
    logger.info("modelo guardado en: %s", path)
    return str(path)


def save_feature_metadata(city_name: str, features: list) -> str:
    """guardar la metadata de las features"""
    folder = get_city_path(city_name, "results")
    path = folder / f"{city_slug(city_name)}_features.json"

    payload = {"total_features": len(features), "features": features}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    logger.info("features guardadas en: %s", path)
    return str(path)
