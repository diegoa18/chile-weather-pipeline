import logging
from datetime import date, timedelta

import pandas as pd
import requests

from src.config.settings import API_URL, ARCHIVE_API_URL, DAILY_VARS
from src.utils.paths import city_slug, get_city_path

logger = logging.getLogger(__name__)

COLUMN_MAP = {
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
    "weathercode": "weather_code",
}


def _build_df_from_api_response(
    data: dict, latitude: float, longitude: float
) -> pd.DataFrame:
    daily = data.get("daily", {})
    if not daily:
        logger.warning("no se encontraron datos 'daily' en la respuesta")
        return pd.DataFrame()

    time_index = daily.get("time", [])
    df_data = {"date": time_index}

    n = len(time_index)
    for api_key, local_key in COLUMN_MAP.items():
        df_data[local_key] = daily.get(api_key, [None] * n)

    df = pd.DataFrame(df_data)
    df["latitude"] = latitude
    df["longitude"] = longitude

    return df


def fetch_historical_data(
    latitude: float, longitude: float, days: int = 365
) -> pd.DataFrame:
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
    }

    try:
        logger.info(
            "descargando datos historicos (%s a %s) para coords (%.2f, %.2f)...",
            start_date,
            end_date,
            latitude,
            longitude,
        )
        resp = requests.get(ARCHIVE_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("error al conectar con archive API: %s", e)
        raise

    return _build_df_from_api_response(data, latitude, longitude)


def fetch_recent_data(latitude: float, longitude: float, days: int = 7) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": days,
        "forecast_days": 0,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
    }

    try:
        logger.info(
            "descargando datos recientes (ultimos %d dias) para coords (%.2f, %.2f)...",
            days,
            latitude,
            longitude,
        )
        resp = requests.get(API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("error al conectar con forecast API: %s", e)
        raise

    return _build_df_from_api_response(data, latitude, longitude)


def save_raw_data(df: pd.DataFrame, city_name: str) -> str:
    folder = get_city_path(city_name, "raw")
    file_path = folder / f"{city_slug(city_name)}_weather_raw.csv"
    df.to_csv(file_path, index=False)
    logger.info("datos crudos guardados en: %s", file_path)
    return str(file_path)
