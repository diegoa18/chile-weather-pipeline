"""
ORQUESTADOR DE LA PIPELINE
clase principal que coordina el flujo de datos: ETL -> analisis -> modelado -> visualizacion
"""

from src.config.settings import get_db_url
#ETL imports
from src.etl.extract import get_weather_data, save_raw_data
from src.etl.transform import load_raw_data, clean_and_transform, save_processed_data
from src.etl.load import init_db_connection, save_to_database
#analysis imports
from src.analysis.metrics import compute_weather_metrics, save_metrics
#modeling imports
from src.modeling.train import prepare_training_data, train_temperature_model, evaluate_model, save_model
from src.modeling.predict import forecast_future, save_forecast
#visualization imports
from src.visualization.plots import plot_temperature_trends, plot_precipitation, plot_forecast

class WeatherPipeline:
    def __init__(self, city_name: str, latitude: float, longitude: float):
        self.city = city_name
        self.latitude = latitude
        self.longitude = longitude
        self.db_engine = init_db_connection()

    def run_etl(self, days_back=30): #toda la fase ETL
        #extraccion
        df_raw = get_weather_data(self.latitude, self.longitude, days=days_back)
        
        if df_raw.empty:
            return None

        save_raw_data(df_raw, self.city)

        #transformacion
        df_raw_loaded = load_raw_data(self.city)
        df_clean = clean_and_transform(df_raw_loaded)
        save_processed_data(df_clean, self.city)

        #carga
        table_name = f"{self.city.lower().replace(' ', '_')}_weather"
        save_to_database(df_clean, table_name, self.db_engine)
        
        return df_clean

    def run_analysis(self, df_clean): #analisis (metricas y visualizaciones)
        metrics = compute_weather_metrics(df_clean)
        save_metrics(self.city, metrics)
        
        #plots
        plot_temperature_trends(df_clean, self.city)
        plot_precipitation(df_clean, self.city)
        
        return metrics

    def run_modeling(self, df_clean): #modelado (entrenamiento y prediccion)
        #entrenamiento
        X, y, features = prepare_training_data(df_clean)
        model = train_temperature_model(X, y)
        metrics = evaluate_model(model, X, y)
        save_model(self.city, model, features)

        #prediccion
        model_payload = {"model": model, "features": features}
        forecast_df = forecast_future(model_payload, df_clean, days_ahead=3)
        save_forecast(self.city, forecast_df, metrics)
        
        #plot prediccion
        plot_forecast(forecast_df, self.city)

        return metrics, forecast_df
