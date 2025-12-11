"""
TRANSFORMACION
"""

import pandas as pd
import numpy as np
from src.utils.paths import get_city_path

def load_raw_data(city_name: str) -> pd.DataFrame:
    """cargar datos crudos desde CSV"""
    folder = get_city_path(city_name, "raw")
    file_path = folder / f"{city_name.lower()}_weather_raw.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"no existe archivo raw en {file_path}")

    return pd.read_csv(file_path)

def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    limpieza y transformacion de datos
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "temp_max", "temp_min", "precipitation", "sunshine_duration",
        "windspeed_10m_max", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", "relative_humidity_max",
        "relative_humidity_min", "dew_point_min", "dew_point_max",
        "cloud_cover_mean"
    ]

    #conversion segura a numerico
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    #filtros de calidad basicos
    df = df.dropna(subset=["temp_max", "temp_min"]).sort_values("date")

    #feature engineering
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]

    #humedad y punto de rocio promedio
    if "relative_humidity_max" in df.columns and "relative_humidity_min" in df.columns:
        df["humidity_avg"] = (df["relative_humidity_max"] + df["relative_humidity_min"]) / 2
    else:
        df["humidity_avg"] = np.nan

    if "dew_point_max" in df.columns and "dew_point_min" in df.columns:
        df["dew_point_avg"] = (df["dew_point_max"] + df["dew_point_min"]) / 2
    else:
        df["dew_point_avg"] = np.nan

    #indice de calor estimado (simplificado)
    #solo valido si tenemos T avg y humedad
    if "humidity_avg" in df.columns:
        #formula simple de heat index para completitud
        df["heat_index_est"] = df["temp_avg"] + 0.1 * df["humidity_avg"]
    else:
        df["heat_index_est"] = np.nan

    #energia solar estimada (kWh/m2 aprox)
    if "shortwave_radiation_sum" in df.columns:
        df["solar_energy"] = df["shortwave_radiation_sum"].fillna(0)
    else:
        df["solar_energy"] = 0

    #imputacion de nulos (media o cero segun logica de negocio)
    cols_to_fill = [
        "precipitation", "sunshine_duration", "windspeed_10m_max",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration",
        "humidity_avg", "dew_point_avg", "cloud_cover_mean",
        "solar_energy", "heat_index_est", "temp_range"
    ]
    
    fill_values = {}
    for col in cols_to_fill:
        if col in df.columns:
            #si la columna tiene valores no nulos, se llena con la media, sino con cero
            fill_values[col] = df[col].mean() if df[col].notna().any() else 0
            
    df = df.fillna(fill_values)

    #redondeo final para limpieza visual
    round_cols = ["temp_max", "temp_min", "temp_avg", "precipitation", "solar_energy"]
    for c in round_cols:
        if c in df.columns:
            df[c] = df[c].round(2)

    return df

def save_processed_data(df: pd.DataFrame, city_name: str) -> str:
    """guardar datos procesados."""
    folder = get_city_path(city_name, "processed")
    file_path = folder / f"{city_name.lower()}_weather_clean.csv"
    df.to_csv(file_path, index=False)
    print(f"datos procesados guardados en: {file_path}")
    return str(file_path)
