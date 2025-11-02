import requests
import pandas as pd
from datetime import date, timedelta
import os
from src.utils.paths import get_city_path

def get_weather_data(latitude, longitude, days=7):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "sunshine_duration",
        "windspeed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "weathercode",
        "relative_humidity_2m_max",
        "relative_humidity_2m_min",
        "dew_point_2m_min",
        "dew_point_2m_max",
        "cloud_cover_mean"
    ]

    URL = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={','.join(daily_vars)}"
        "&timezone=auto"
    )

    print(f"solicitando datos desde {URL}")
    resp = requests.get(URL)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    n = len(daily.get("time", []))

    df = pd.DataFrame({
        "date": daily.get("time", []),
        "temp_max": daily.get("temperature_2m_max", [None]*n),
        "temp_min": daily.get("temperature_2m_min", [None]*n),
        "precipitation": daily.get("precipitation_sum", [None]*n),
        "sunshine_duration": daily.get("sunshine_duration", [None]*n),
        "windspeed_10m_max": daily.get("windspeed_10m_max", [None]*n),
        "shortwave_radiation_sum": daily.get("shortwave_radiation_sum", [None]*n),
        "et0_fao_evapotranspiration": daily.get("et0_fao_evapotranspiration", [None]*n),
        "relative_humidity_max": daily.get("relative_humidity_2m_max", [None]*n),
        "relative_humidity_min": daily.get("relative_humidity_2m_min", [None]*n),
        "dew_point_min": daily.get("dew_point_2m_min", [None]*n),
        "dew_point_max": daily.get("dew_point_2m_max", [None]*n),
        "cloud_cover_mean": daily.get("cloud_cover_mean", [None]*n),
        "weather_code": daily.get("weathercode", [None]*n),
    })

    df["latitude"] = latitude
    df["longitude"] = longitude

    return df


def save_weather_data(df, city_name):
    folder = get_city_path(city_name, "raw")
    file_path = os.path.join(folder, f"{city_name.lower()}_weather_raw.csv")
    df.to_csv(file_path, index=False)