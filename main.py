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
import pandas as pd

def main():
    cities = {
        "Temuco": (-38.7397, -72.5984),
        "Santiago": (-33.4569, -70.6483),
        "Valparaiso": (-33.0458, -71.6197),
        "Antofagasta": (-23.6500, -70.4000),
        "Punta Arenas": (-53.1638, -70.9171)
    }

    engine = init_db()

    for city, (lat, lon) in cities.items():
        print(f"procesando datos de {city}...")

        df_raw = get_weather_data(lat, lon, days=14)
        save_weather_data(df_raw, city)

        df_clean = clean_and_transform(df_raw)
        save_processed_data(df_clean, city)

        df_clean["city"] = city
        df_clean["latitude"] = lat
        df_clean["longitude"] = lon

        save_to_database(df_clean, "weather_data", engine)

        metrics = compute_weather_metrics(df_clean)
        metrics_path = save_metrics(city, metrics)

        x, y, df_train = prepare_training_data(df_clean)
        model = train_temperature_model(x, y)
        model_path = save_model(city, model)

        model_matrics = evaluate_model(model, x, y)
        forecast_df = forecast_future(model, df_train, days_ahead=5)
        save_forecast(city, forecast_df, model_matrics)

        print(f"{city} ha sido procesada :v\n")

