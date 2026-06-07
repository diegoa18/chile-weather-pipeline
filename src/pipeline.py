"""
ORQUESTADOR DE LA PIPELINE
clase principal que coordina el flujo de datos: ETL -> analisis -> modelado -> visualizacion
"""

import logging

import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.analysis.exploratory import residual_analysis, run_full_eda, save_eda_report
from src.analysis.importance import run_feature_importance, save_importance_report
from src.analysis.metrics import compute_weather_metrics, save_metrics
from src.config.settings import DEFAULT_FORECAST_DAYS
from src.etl.extract import fetch_historical_data, save_raw_data
from src.etl.load import init_db_connection, save_to_database
from src.etl.transform import clean_and_transform, load_raw_data, save_processed_data
from src.modeling.predict import forecast_future, save_forecast
from src.modeling.train import (
    evaluate_model,
    prepare_training_data,
    save_feature_metadata,
    save_model,
    train_temperature_model,
)
from src.utils.paths import city_slug
from src.visualization.plots import (
    plot_forecast,
    plot_precipitation,
    plot_temperature_trends,
)

logger = logging.getLogger(__name__)


class WeatherPipeline:
    def __init__(self, city_name: str, latitude: float, longitude: float):
        self.city = city_name
        self.latitude = latitude
        self.longitude = longitude
        self.db_engine = init_db_connection()

    def run_etl(self, days_back=365):
        df_raw = fetch_historical_data(self.latitude, self.longitude, days=days_back)
        if df_raw.empty:
            return None

        save_raw_data(df_raw, self.city)
        df_raw_loaded = load_raw_data(self.city)
        df_clean = clean_and_transform(df_raw_loaded)
        save_processed_data(df_clean, self.city)

        table_name = f"{city_slug(self.city)}_weather"
        save_to_database(df_clean, table_name, self.db_engine)
        return df_clean

    def run_analysis(self, df_clean):
        metrics = compute_weather_metrics(df_clean)
        save_metrics(self.city, metrics)

        eda = run_full_eda(df_clean, self.city)
        save_eda_report(eda, self.city)

        plot_temperature_trends(df_clean, self.city)
        plot_precipitation(df_clean, self.city)
        return metrics, eda

    def run_modeling(self, df_clean, days_ahead=DEFAULT_FORECAST_DAYS):
        X, y, features = prepare_training_data(df_clean)

        model = train_temperature_model(X, y)
        metrics, y_true, y_pred = evaluate_model(X, y)

        save_model(self.city, model, features)
        save_feature_metadata(self.city, features)

        # feature importance
        importance = run_feature_importance(model, X, y)
        save_importance_report(importance, self.city)

        residual_report = residual_analysis(pd.Series(y_true), pd.Series(y_pred))
        logger.info("analisis de residuos: %s", residual_report)

        # baseline persistence
        y_baseline = y.shift(1).dropna()
        y_actual = y.loc[y_baseline.index]
        baseline_mae = mean_absolute_error(y_actual, y_baseline)
        logger.info(
            "baseline persistence MAE: %.2f (vs modelo MAE: %.2f)",
            baseline_mae,
            metrics["MAE"],
        )

        metrics["baseline_persistence_MAE"] = round(float(baseline_mae), 2)
        metrics["residuals"] = residual_report

        # forecast recursivo
        model_payload = {"model": model, "features": features}
        forecast_df = forecast_future(model_payload, df_clean, days_ahead=days_ahead)
        save_forecast(self.city, forecast_df, metrics)
        plot_forecast(forecast_df, self.city)

        return metrics, forecast_df
