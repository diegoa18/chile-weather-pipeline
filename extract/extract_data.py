import requests
import pandas as pd
from datetime import date, timedelta
import os

def get_weather_data(latitude, longitude, days=7):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    URL = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=auto"
    )

    print(f"solicitando datos desde {URL}")
    response = requests.get(URL)

    if response.status_code != 200:
        raise Exception(f"error en la API: {response.status_code}")
    
    data = response.json()

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "precipitation": data["daily"]["precipitation_sum"]
    })

    df["latitude"] = latitude
    df["longitude"] = longitude
    return df

def save_weather_data(df, city_name):
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{city_name.lower()}_weather_raw.csv"
    df.to_csv(file_path, index=False)
    print(f"datos guardados en {file_path}")