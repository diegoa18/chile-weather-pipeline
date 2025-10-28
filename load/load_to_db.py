import pandas as pd
import sqlite3
import os

def load_to_db(city_name):
    file_path = f"data/processed/{city_name.lower()}_weather_clean.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"no existe el archivo procesado: {file_path}")
    
    df = pd.read_csv(file_path)

    os.makedirs("data/db", exist_ok=True)
    connect = sqlite3.connect("data/db/weather_data.db")

    table_name = f"{city_name.lower()}_weather"
    df.to_sql(table_name, connect, if_exists="replace", index=False)

    connect.close()