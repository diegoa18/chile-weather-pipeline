import logging
from matplotlib.dates import DateFormatter, MonthLocator

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.paths import get_city_path

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")


def plot_temperature_trends(df: pd.DataFrame, city_name: str) -> None:
    if df.empty:
        logger.warning("no hay datos para graficar tendencias en %s", city_name)
        return

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["temp_max"], label="Max Temp", color="#d62728", linewidth=2)
    ax.plot(df["date"], df["temp_min"], label="Min Temp", color="#1f77b4", linewidth=2)
    ax.plot(df["date"], df["temp_avg"], label="Avg Temp", color="#ff7f0e", linestyle="--")

    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

    ax.set_title(f"Tendencia de Temperaturas - {city_name}", fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Temperatura (°C)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    folder = get_city_path(city_name, "plots")
    path = folder / "temperature_trend.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("grafico guardado: %s", path)


def plot_precipitation(df: pd.DataFrame, city_name: str) -> None:
    if df.empty:
        return

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["date"], df["precipitation"], width=0.8, color="#17becf")

    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

    ax.set_title(f"Precipitación Diaria - {city_name}", fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precipitación (mm)")
    fig.tight_layout()

    folder = get_city_path(city_name, "plots")
    path = folder / "precipitation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("grafico guardado: %s", path)


def plot_forecast(
    forecast_df: pd.DataFrame,
    city_name: str,
    history: pd.DataFrame | None = None,
) -> None:
    if forecast_df.empty:
        return

    if not pd.api.types.is_datetime64_any_dtype(forecast_df["date"]):
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    fig, ax = plt.subplots(figsize=(8, 4))

    if history is not None and "temp_avg" in history.columns:
        if not pd.api.types.is_datetime64_any_dtype(history["date"]):
            history["date"] = pd.to_datetime(history["date"])
        window = history.sort_values("date").tail(30)
        ax.plot(
            window["date"], window["temp_avg"],
            label="Histórico (30d)", color="#1f77b4", linewidth=1.5,
        )

    ax.plot(
        forecast_df["date"], forecast_df["predicted_temp_avg"],
        marker="o", color="#2ca02c", linewidth=2, label="Pronóstico",
    )

    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))

    ax.set_title(f"Pronóstico Temperatura Promedio - {city_name}", fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Temp. (°C)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    folder = get_city_path(city_name, "plots")
    path = folder / "forecast.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("grafico guardado: %s", path)
