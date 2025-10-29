import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
import os

def prepare_training_data(df):

    if df.empty:
        raise ValueError("el DF esta vacio")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["temp_avg"])

    df["day_number"] = (df["date"] - df["date"].min()).dt.days

    x = df[["day_number"]]
    y = df["temp_avg"]

    return x, y, df

def train_temperature_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model

def evaluate_model(model, x, y):
    preds = model.predict(x)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5
    return{"MAE": round(mae, 3), "RMSE": round(rmse, 3)}

def save_model(city_name, model):
    os.makedirs("analysis/models", exist_ok=True)
    path = f"analysis/models/{city_name.lower()}_temp_model.pkl"
    joblib.dump(model, path)
    return path

def forecast_future(model, df, days_ahead=7):
    last_day = df["day_number"].max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)

    preds = model.predict(future_days).ravel()  

    future_dates = pd.date_range(
        start=df["date"].max() + pd.Timedelta(days=1),
        periods=days_ahead
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_temp_avg": preds.round(2)
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


