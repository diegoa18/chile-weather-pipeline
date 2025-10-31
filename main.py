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

CITY_NAME = "Temuco"
LATITUDE = -38.7366
LONGITUDE = -72.595
DAYS_HISTORY = 30  # más días mejora el training

def main():
    print("=== INICIANDO PIPELINE CLIMÁTICA ===")

    # 1. Extracción
    df_raw = get_weather_data(LATITUDE, LONGITUDE, DAYS_HISTORY)
    save_weather_data(df_raw, CITY_NAME)

    # 2. Transformación
    df_raw_loaded = load_raw_data(CITY_NAME)
    df_clean = clean_and_transform(df_raw_loaded)
    save_processed_data(df_clean, CITY_NAME)

    # 3. Análisis descriptivo
    df_loaded = load_clean_data(CITY_NAME)
    metrics = compute_weather_metrics(df_loaded)
    metrics_path = save_metrics(CITY_NAME, metrics)
    print(f"✅ Métricas guardadas en {metrics_path}")

    # 4. Preparar datos y entrenar
    X, y, df_train = prepare_training_data(df_loaded)
    model = train_temperature_model(X, y)

    # 5. Guardar modelo con features
    model_path = save_model(CITY_NAME, model, X.columns.tolist())
    print(f"✅ Modelo entrenado y guardado en {model_path}")

    # 6. Evaluar
    model_metrics = evaluate_model(model, X, y)
    print(f"📊 Métricas del modelo: {model_metrics}")

    # 7. Pronosticar
    forecast_df = forecast_future({"model": model, "features": X.columns.tolist()}, df_train, days_ahead=5)
    forecast_path, model_metrics_path = save_forecast(CITY_NAME, forecast_df, model_metrics)
    print(f"🌤️ Pronóstico guardado en {forecast_path}")

    # 8. Persistir datos en DB
    engine = init_db()
    save_to_database(df_clean, f"{CITY_NAME.lower()}_weather", engine)
    print("💾 Datos guardados en base local SQLite")

    print("\n✅ Pipeline completado con éxito.")

if __name__ == "__main__":
    main()
