import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from src.utils.paths import get_city_path
from src.config.settings import MODEL_PARAMS_XGB, MODEL_PARAMS_RF

#intento de importar XGBoost, fallback a Random Forest
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

    #seleccion de features potenciales
    feature_cols = [
        "precipitation",
        "windspeed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "humidity_avg",
        "dew_point_avg",
        "cloud_cover_mean",
        "solar_energy",
        "temp_range"
    ]

    #filtrar solo las columnas que existen en el DF
    available_features = [c for c in feature_cols if c in df.columns]
    
    if not available_features:
        raise ValueError("no hay features disponibles para entrenar el modelo")

    #relleno de nulos simple para asegurar integridad
    X = df[available_features].fillna(0)
    y = df["temp_avg"]

    return X, y, available_features

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    
    model.fit(X_train, y_train) #re-entrenamiento en train set para evaluación honesta
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    
    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}
    print(f"metricas del modelo: {metrics}")
    
    #re-entrenar con todo el dataset para producción
    model.fit(X, y)
    
    return metrics

def save_model(city_name: str, model, features: list) -> str:
    """guardar el modelo entrenado y los nombres de las features."""
    folder = get_city_path(city_name, "models")
    path = folder / f"{city_name.lower()}_temp_model.pkl"

    payload = {
        "model": model,
        "features": features,
        "model_type": "xgboost" if USE_XGB else "random_forest"
    }

    joblib.dump(payload, path)
    print(f"modelo guardado en: {path}")
    return str(path)
