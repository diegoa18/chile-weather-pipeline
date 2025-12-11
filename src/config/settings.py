"""
CONFIGURACION
"""

import os
from pathlib import Path

#definición de rutas base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

#configuración de API Open-Meteo
API_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = [
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

#Configuracion de modelos
MODEL_PARAMS_XGB = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0
}

MODEL_PARAMS_RF = {
    "n_estimators": 200,
    "max_depth": 12,
    "random_state": 42
}

def get_db_url():
    return os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/weather_data.db")
