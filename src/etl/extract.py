"""
se encarga de obtener datos crudos desde la API de Open-Meteo
"""

import requests
import pandas as pd
from datetime import date, timedelta
from src.utils.paths import get_city_path
from src.config.settings import API_URL, DAILY_VARS

def get_weather_data(latitude: float, longitude: float, days: int = 7) -> pd.DataFrame:
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto"
    }

    try:
        print(f"solicitando datos a {API_URL} para coords ({latitude}, {longitude})...")
        resp = requests.get(API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"error al conectar con la API: {e}")
        raise

    daily = data.get("daily", {})
    if not daily:
        print("advertencia: no se encontraron datos 'daily' en la respuesta")
        return pd.DataFrame()

    #construccion del DF
    time_index = daily.get("time", [])
    df_data = {"date": time_index}

    #mapeo de claves de API a nombres de columnas locales
    column_map = {
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precipitation",
        "sunshine_duration": "sunshine_duration",
        "windspeed_10m_max": "windspeed_10m_max",
        "shortwave_radiation_sum": "shortwave_radiation_sum",
        "et0_fao_evapotranspiration": "et0_fao_evapotranspiration",
        "relative_humidity_2m_max": "relative_humidity_max",
        "relative_humidity_2m_min": "relative_humidity_min",
        "dew_point_2m_min": "dew_point_min",
        "dew_point_2m_max": "dew_point_max",
        "cloud_cover_mean": "cloud_cover_mean",
        "weathercode": "weather_code"
    }

    n = len(time_index)
    for api_key, local_key in column_map.items():
        df_data[local_key] = daily.get(api_key, [None] * n)

    df = pd.DataFrame(df_data)
    df["latitude"] = latitude
    df["longitude"] = longitude

    return df

def save_raw_data(df: pd.DataFrame, city_name: str) -> str:
    """
    guardar los datos crudos en formato CSV
    """
    folder = get_city_path(city_name, "raw")
    file_path = folder / f"{city_name.lower()}_weather_raw.csv"
    df.to_csv(file_path, index=False)
    print(f"datos crudos guardados en: {file_path}")
    return str(file_path)
