# main.py
import os
from extract.extract_data import get_weather_data, save_weather_data
from transform.transform_data import load_raw_data, clean_and_transform, save_processed_data
from analysis.analyze_data import load_clean_data, compute_weather_metrics, save_metrics
from analysis.model_temperature import (
    prepare_training_data,
    train_temperature_model,
    evaluate_model,
    save_model,
    forecast_future,
    save_forecast
)
from load.load_data import init_db, save_to_database