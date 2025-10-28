import pandas as pd
import os

def load_raw_data(city_name):
    file_path = f"data/raw/{city_name.lower()}_weather_raw.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"no existe el archivo {file_path}")
    return pd.read_csv(file_path)

def clean_and_transform(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["temp_max", "temp_min"])

    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2

    df["temp_max"] = df["temp_max"].round(1)
    df["temp_min"] = df["temp_min"].round(1)
    df["temp_avg"] = df["temp_avg"].round(1)
    df["precipitation"] = df["precipitation"].round(2)

    df = df.sort_values(by="date")

    return df

def save_processed_data(df, city_name):
    os.makedirs("data/processed", exist_ok=True)
    file_path = f"data/processed/{city_name.lower()}_weather_clean.csv"
    df.to_csv(file_path, index=False)
    print(f"datos limpios guardados en {file_path}")