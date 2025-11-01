import os
from src.extract.extract_data import get_weather_data, save_weather_data
from src.transform.transform_data import load_raw_data, clean_and_transform, save_processed_data
from src.analysis.analyze_data import load_clean_data, compute_weather_metrics, save_metrics
from src.analysis.model_temperature import (
    prepare_training_data,
    train_temperature_model,
    evaluate_model,
    save_model,
    forecast_future,
    save_forecast
)
from src.load.load_data import init_db, save_to_database


class WeatherPipeline:
    def __init__(self, city_name, latitude, longitude, db_url=None):
        self.city = city_name
        self.latitude = latitude
        self.longitude = longitude
        self.db_engine = init_db(db_url or "data/weather_data.db")

    def extract(self, days=7):
        df = get_weather_data(self.latitude, self.longitude, days)
        save_weather_data(df, self.city)
        return df
    
    def transform(self):
        df_raw = load_raw_data(self.city)
        df_clean = clean_and_transform(df_raw)
        save_processed_data(df_clean, self.city)
        save_to_database(df_clean, f"{self.city.lower()}_weather", self.db_engine)
        return df_clean
    
    def analyze(self, df_clean):
        metrics = compute_weather_metrics(df_clean)
        save_metrics(self.city, metrics)
        return metrics
    
    def model(self, df_clean):
        X, y, df_meta = prepare_training_data(df_clean)
        model = train_temperature_model(X, y)
        metrics = evaluate_model(model, X, y)
        save_model(self.city, model, df_meta.attrs["used_features"])
        forecast_df = forecast_future(model, df_meta, days_ahead=3)
        save_forecast(self.city, forecast_df, metrics)
        return metrics, forecast_df
