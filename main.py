import argparse
import logging

from src.config.logger import setup_logging
from src.config.settings import CITIES, DEFAULT_DAYS_BACK, DEFAULT_FORECAST_DAYS
from src.modeling.predict import (
    forecast_future,
    load_features,
    load_model,
    save_forecast,
)
from src.pipeline import WeatherPipeline, load_clean_data
from src.visualization.plots import plot_forecast

setup_logging()
logger = logging.getLogger(__name__)


def cmd_train():
    for city, (lat, lon) in CITIES.items():
        try:
            logger.info("=" * 50)
            logger.info("Entrenando: %s", city)
            logger.info("=" * 50)

            pipeline = WeatherPipeline(city, lat, lon)
            df_clean = pipeline.run_etl(days_back=DEFAULT_DAYS_BACK)

            if df_clean is not None:
                pipeline.run_analysis(df_clean)
                pipeline.run_modeling(df_clean)

        except Exception as e:
            logger.exception("Error procesando %s: %s", city, e)
            continue


def cmd_predict():
    for city, (lat, lon) in CITIES.items():
        try:
            logger.info("=" * 50)
            logger.info("pronosticando: %s", city)
            logger.info("=" * 50)

            model = load_model(city)
            features = load_features(city)
            df_clean = load_clean_data(city)

            model_payload = {"model": model, "features": features}
            forecast_df = forecast_future(
                model_payload, df_clean, days_ahead=DEFAULT_FORECAST_DAYS
            )
            save_forecast(city, forecast_df)
            plot_forecast(forecast_df, city, history=df_clean)

        except Exception as e:
            logger.exception("error pronosticando %s: %s", city, e)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline clima Chile")
    parser.add_argument(
        "mode",
        choices=["train", "predict"],
        help="train: ETL + entrena modelo + guarda metricas | predict: forecast con modelo guardado",
    )
    args = parser.parse_args()

    if args.mode == "train":
        cmd_train()
    else:
        cmd_predict()
