import json

import joblib
import numpy as np
import pandas as pd

from src.utils.paths import get_city_path


def load_model(city_name: str):  # carga un modelo entrenado
    folder = get_city_path(city_name, "models")
    path = folder / f"{city_name.lower()}_temp_model.pkl"

    if not path.exists():
        raise FileNotFoundError(f"no se encontro modelo para {city_name} en {path}")

    return joblib.load(path)


# genera pronostico futuro simulando condiciones climaticas
def forecast_future(
    model_payload: dict, recent_data: pd.DataFrame, days_ahead: int = 3
) -> pd.DataFrame:
    model = model_payload["model"]
    features = model_payload["features"]

    missing_features = [f for f in features if f not in recent_data.columns]

    if missing_features:
        raise NotImplementedError(
            "forecasting con features temporales aun no implementado. "
            "Se requiere generar lags y rolling de forma recursiva."
        )

    # tomar el promedio de los ultimos dias como base
    window_size = 5
    recent_mean = recent_data[features].tail(window_size).mean()

    # random noise
    future_features = []
    for _ in range(days_ahead):
        noise = np.random.normal(0, 0.05, size=len(features))
        future_features.append(recent_mean.values * (1 + noise))

    X_future = pd.DataFrame(future_features, columns=features)

    preds = model.predict(X_future)

    last_date = pd.to_datetime(recent_data["date"].max())
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=days_ahead
    )

    forecast_df = pd.DataFrame(
        {"date": future_dates, "predicted_temp_avg": np.round(preds, 2)}
    )

    return forecast_df


# guardar pronostico
def save_forecast(city_name: str, forecast_df: pd.DataFrame, metrics: dict) -> str:
    folder = get_city_path(city_name, "results")

    # guardar csv
    csv_path = folder / f"{city_name.lower()}_forecast.csv"
    forecast_df.to_csv(csv_path, index=False)

    # guardar metadata/metricas
    json_path = folder / f"{city_name.lower()}_model_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"pronostico guardado en: {csv_path}")
    return str(csv_path)
