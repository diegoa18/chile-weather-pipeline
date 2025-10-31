import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
import os
try:
    from xgboost import XGBRegressor
    MODEL_CLASS = "xgb"
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    MODEL_CLASS = "rf"

def prepare_training_data(df):

    if df.empty:
        raise ValueError("el DF esta vacio")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    feature_cols = [
        "precipitation",
        "windspeed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "humidity_avg",
        "dew_point_avg",
        "cloud_cover_mean",
        "surface_pressure",
        "solar_energy",
        "temp_range"
    ]


    available_features = [c for c in feature_cols if c in df.columns]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        print(f"las columnas que faltan son {missing_features}")

    X = df[available_features].copy()
    y = df["temp_avg"].copy()

    X = X.fillna(X.mean())

    df.attrs = dict(df.attrs)
    df.attrs["used_features"] = available_features

    return X, y, df


def train_temperature_model(X, y):
    if MODEL_CLASS == "xgb":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)

    model.fit(X, y)
    return model


def evaluate_model(model, x, y):
    preds = model.predict(x)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5
    return{"MAE": round(mae, 2), "RMSE": round(rmse, 2)}

def save_model(city_name, model, features):
    os.makedirs("analysis/models", exist_ok=True)
    path = f"analysis/models/{city_name.lower()}_temp_model.pkl"
    joblib.dump({"model": model, "features": features, "model_class": MODEL_CLASS}, path)
    return path

def forecast_future(model_obj, df, days_ahead=3):
    if isinstance(model_obj, dict):
        model = model_obj.get("model")
        used_features = model_obj.get("features", [])
    else:
        model = model_obj
        used_features = df.attrs.get("used_features", [])

    if not used_features:
        raise ValueError("no hay features usadas para la prediccion")


    last_row = df.iloc[-1]
    base = last_row[used_features]
    X_future = pd.DataFrame([base.values for _ in range(days_ahead)], columns=used_features)

    for col in used_features:
        if col not in X_future.columns:
            X_future[col] = 0

    X_future = X_future.fillna(X_future.mean())


    preds = model.predict(X_future)
    future_dates = pd.date_range(start=df["date"].max() + pd.Timedelta(days=1), periods=days_ahead)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_temp_avg": np.round(preds, 2)
    })

    
    return forecast_df


def save_forecast(city_name, forecast_df, metrics):
    os.makedirs("analysis/results", exist_ok=True)

    forecast_path = f"analysis/results/{city_name.lower()}_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    metrics_path = f"analysis/results/{city_name.lower()}_model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    return forecast_path, metrics_path


