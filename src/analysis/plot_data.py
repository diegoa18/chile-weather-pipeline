import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.paths import get_city_path

sns.set(style="whitegrid")

def plot_temperature_trends(city_name):
    folder = get_city_path(city_name, "processed")
    file_path = os.path.join(folder, f"{city_name.lower()}_weather_clean.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"no existe el archivo limpio {file_path}")
    
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["temp_max"], label="temp max", color="red")
    plt.plot(df["date"], df["temp_min"], label="temp min", color="blue")
    plt.plot(df["date"], df["temp_avg"], label="temp promedio", color="orange", linestyle="--")

    plt.title(f"tendencia de temperaturas - {city_name}")
    plt.xlabel("fecha")
    plt.ylabel("temperatura (C)")
    plt.legend()
    plt.tight_layout()

    plot_folder = get_city_path(city_name, "plots")
    plt.savefig(os.path.join(plot_folder, "temperature_trend.png"))
    plt.close()


def plot_precipitation(city_name):
    folder = get_city_path(city_name, "processed")
    file_path = os.path.join(folder, f"{city_name.lower()}_weather_clean.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"no existe el archivo limpio {file_path}")

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    plt.figure(figsize=(10, 4))
    sns.barplot(x="date", y="precipitation", data=df, color="skyblue")
    plt.xticks(rotation=45)
    plt.title(f"precipitacion diaria - {city_name}")
    plt.xlabel("fecha")
    plt.ylabel("precipitacion (mm)")
    plt.tight_layout()

    plot_folder = get_city_path(city_name, "plots")
    plt.savefig(os.path.join(plot_folder, "precipitation.png"))
    plt.close()


def plot_forecast(city_name):
    folder = get_city_path(city_name, "results")
    forecast_path = os.path.join(folder, f"{city_name.lower()}_forecast.csv")

    if not os.path.exists(forecast_path):
        raise FileNotFoundError(f"no existe el archivo: {forecast_path}")

    forecast_df = pd.read_csv(forecast_path)
    forecast_df["date"] = pd.to_datetime(forecast_df["date"], errors="coerce")

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=forecast_df, x="date", y="predicted_temp_avg", marker="o", color="green")
    plt.title(f"pronostico de temperatura promedio - {city_name}")
    plt.xlabel("fecha")
    plt.ylabel("temperatura (C)")
    plt.tight_layout()

    plot_folder = get_city_path(city_name, "plots")
    plt.savefig(os.path.join(plot_folder, "forecast.png"))
    plt.close()
