"""
ENTRY POINT PRINCIPAL
ejecuta la pipeline completa para ciudades chilenas
"""

import logging

from src.config.logger import setup_logging
from src.config.settings import CITIES, DEFAULT_DAYS_BACK
from src.pipeline import WeatherPipeline

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    for city, (lat, lon) in CITIES.items():
        try:
            logger.info("=" * 50)
            logger.info("Procesando ciudad: %s", city)
            logger.info("=" * 50)

            pipeline = WeatherPipeline(city, lat, lon)

            df_clean = pipeline.run_etl(days_back=DEFAULT_DAYS_BACK)

            if df_clean is not None:
                pipeline.run_analysis(df_clean)
                pipeline.run_modeling(df_clean)

        except Exception as e:
            logger.exception("Error procesando %s: %s", city, e)
            continue