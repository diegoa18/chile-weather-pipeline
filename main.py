from extract.extract_data import get_weather_data, save_weather_data
from transform.transform_data import clean_and_transform, save_processed_data
from load.load_to_db import load_to_db

def run_pipeline(city_name, latitude, longitude, days=7):
    df_raw = get_weather_data(latitude, longitude, days)
    save_weather_data(df_raw, city_name)

    df_clean = clean_and_transform(df_raw)
    save_processed_data(df_clean, city_name)

    load_to_db(city_name)


if __name__ == "__main__":
    cities = {
        "Temuco": (-38.7397, -72.5984),
        "Santiago": (-33.4569, -70.6483),
        "Valdivia": (-39.8196, -73.2459),
    }

    for city, coords in cities.items():
        run_pipeline(city, coords[0], coords[1], days=7)