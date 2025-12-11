import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils.paths import get_city_path

#estilo global
sns.set(style="whitegrid")

def plot_temperature_trends(df: pd.DataFrame, city_name: str) -> None:
    if df.empty:
        print(f"no hay datos para graficar tendencias en {city_name}")
        return

    #asegurar fechas
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["temp_max"], label="Max Temp", color="#d62728", linewidth=2)
    plt.plot(df["date"], df["temp_min"], label="Min Temp", color="#1f77b4", linewidth=2)
    plt.plot(df["date"], df["temp_avg"], label="Avg Temp", color="#ff7f0e", linestyle="--")

    plt.title(f"Tendencia de Temperaturas - {city_name}", fontsize=14)
    plt.xlabel("Fecha")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    #guardar
    folder = get_city_path(city_name, "plots")
    path = folder / "temperature_trend.png"
    plt.savefig(path)
    plt.close()
    print(f"grafico guardado: {path}")

#grafico precipitaciones
def plot_precipitation(df: pd.DataFrame, city_name: str) -> None:
    if df.empty:
        return

    plt.figure(figsize=(10, 4))
    sns.barplot(x=df["date"].dt.strftime("%Y-%m-%d"), y=df["precipitation"], color="#17becf")
    
    plt.xticks(rotation=45)
    plt.title(f"Precipitación Diaria - {city_name}", fontsize=14)
    plt.xlabel("Fecha")
    plt.ylabel("Precipitación (mm)")
    plt.tight_layout()

    folder = get_city_path(city_name, "plots")
    path = folder / "precipitation.png"
    plt.savefig(path)
    plt.close()
    print(f"grafico guardado: {path}")

#grafico prediccion
def plot_forecast(forecast_df: pd.DataFrame, city_name: str) -> None:
    if forecast_df.empty:
        return

    #asegurar fechas
    if not pd.api.types.is_datetime64_any_dtype(forecast_df["date"]):
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=forecast_df, 
        x="date", 
        y="predicted_temp_avg", 
        marker="o", 
        color="#2ca02c",
        linewidth=2
    )
    
    plt.title(f"Pronóstico Temperatura Promedio - {city_name}", fontsize=14)
    plt.xlabel("Fecha Futura")
    plt.ylabel("Temp. Predicha (°C)")
    plt.grid(True)
    plt.tight_layout()

    folder = get_city_path(city_name, "plots")
    path = folder / "forecast.png"
    plt.savefig(path)
    plt.close()
    print(f"grafico guardado: {path}")
