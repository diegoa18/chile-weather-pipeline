"""
ENTRY POINT TEMPORAL, EL PIPELINE SE TESTEA CON X CIUDADES
"""

from src.pipeline import WeatherPipeline

if __name__ == "__main__":
    #ciudades objetivo
    CHILE_CITIES = {
        "Santiago": (-33.45, -70.66),
        "Concepcion": (-36.82, -73.05),
        "Puerto Montt": (-41.47, -72.94),
        "Antofagasta": (-23.65, -70.40) #antofa pa variar lol
    }

    for city, (lat, lon) in CHILE_CITIES.items():
        try:
            pipeline = WeatherPipeline(city, lat, lon)
            
            #ETL
            df_clean = pipeline.run_etl(days_back=30)
            
            if df_clean is not None:
                #analisis
                pipeline.run_analysis(df_clean)
                
                #modelado
                pipeline.run_modeling(df_clean)
            
        except Exception as e:
            print(f"[ERROR] {city}: {e}")
            #continuamos con la siguiente ciudad
            continue