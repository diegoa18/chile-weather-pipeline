import logging

import numpy as np
import pandas as pd

from src.utils.paths import city_slug, get_city_path

logger = logging.getLogger(__name__)


def load_raw_data(city_name: str) -> pd.DataFrame:
    folder = get_city_path(city_name, "raw")
    file_path = folder / f"{city_slug(city_name)}_weather_raw.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"no existe archivo raw en {file_path}")

    return pd.read_csv(file_path)


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "temp_max",
        "temp_min",
        "precipitation",
        "sunshine_duration",
        "windspeed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "relative_humidity_max",
        "relative_humidity_min",
        "dew_point_min",
        "dew_point_max",
        "cloud_cover_mean",
    ]

    # conversion segura a numerico
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # filtros de calidad basicos
    df = df.dropna(subset=["temp_max", "temp_min"]).sort_values("date")

    # feature engineering
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]

    # humedad y punto de rocio promedio
    if "relative_humidity_max" in df.columns and "relative_humidity_min" in df.columns:
        df["humidity_avg"] = (
            df["relative_humidity_max"] + df["relative_humidity_min"]
        ) / 2
    else:
        df["humidity_avg"] = np.nan

    if "dew_point_max" in df.columns and "dew_point_min" in df.columns:
        df["dew_point_avg"] = (df["dew_point_max"] + df["dew_point_min"]) / 2
    else:
        df["dew_point_avg"] = np.nan

    # imputacion de nulos (media cuando hay datos, cero si no)
    cols_to_fill = [
        "precipitation",
        "sunshine_duration",
        "windspeed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "humidity_avg",
        "dew_point_avg",
        "cloud_cover_mean",
        "temp_range",
    ]

    fill_values = {}
    for col in cols_to_fill:
        if col in df.columns:
            # si la columna tiene valores no nulos, se llena con la media, sino con cero
            fill_values[col] = df[col].mean() if df[col].notna().any() else 0

    df = df.fillna(fill_values)

    # redondeo final para limpieza visual
    round_cols = ["temp_max", "temp_min", "temp_avg", "precipitation"]
    for c in round_cols:
        if c in df.columns:
            df[c] = df[c].round(2)

    return df


def save_processed_data(df: pd.DataFrame, city_name: str) -> str:
    folder = get_city_path(city_name, "processed")
    file_path = folder / f"{city_slug(city_name)}_weather_clean.csv"
    df.to_csv(file_path, index=False)
    logger.info("datos procesados guardados en: %s", file_path)
    return str(file_path)
