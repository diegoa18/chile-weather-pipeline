import pandas as pd
import json
import os

def load_clean_data(city_name):
    file_path = f"data/processed/{city_name.lower()}_weather_clean.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"no existe el archivo limpio {file_path}")
    return pd.read_csv(file_path)

def compute_weather_metrics(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    metrics = {
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date())
        },
        "temperature": {
            "avg": round(df["temp_avg"].mean(), 2),
            "max_mean": round(df["temp_max"].mean(), 2),
            "min_mean": round(df["temp_min"].mean(), 2),
            "range_avg": round(df["temp_range"].mean(), 2) if "temp_range" in df.columns else None
        },
        "precipitation": {
            "total": round(df["precipitation"].sum(), 2),
            "avg_daily": round(df["precipitation"].mean(), 2)
        },
        "humidity": {
            "avg": round(df["humidity_avg"].mean(), 2) if "humidity_avg" in df.columns else None,
            "dew_point_avg": round(df["dew_point_avg"].mean(), 2) if "dew_point_avg" in df.columns else None
        },
        "solar": {
            "energy_avg": round(df["solar_energy"].mean(), 2) if "solar_energy" in df.columns else None
        },
        "records": len(df)
    }
    return metrics


def save_metrics(city_name, metrics):
    os.makedirs("data/results", exist_ok=True)
    file_path = f"data/results/{city_name.lower()}_metrics.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    return file_path