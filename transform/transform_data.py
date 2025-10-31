import pandas as pd
import os

def load_raw_data(city_name):
    file_path = f"data/raw/{city_name.lower()}_weather_raw.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"no existe {file_path}")
    return pd.read_csv(file_path)

def clean_and_transform(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "temp_max", "temp_min", "precipitation", "sunshine_duration",
        "windspeed_10m_max", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", "relative_humidity_max",
        "relative_humidity_min", "surface_pressure",
        "dew_point_min", "dew_point_max", "cloud_cover_mean"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["temp_max", "temp_min"]).sort_values("date")

    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]

    if "relative_humidity_max" in df.columns and "relative_humidity_min" in df.columns:
        df["humidity_avg"] = (df["relative_humidity_max"] + df["relative_humidity_min"]) / 2
    else:
        df["humidity_avg"] = None

    if "dew_point_max" in df.columns and "dew_point_min" in df.columns:
        df["dew_point_avg"] = (df["dew_point_max"] + df["dew_point_min"]) / 2
    else:
        df["dew_point_avg"] = None

    if "humidity_avg" in df.columns:
        df["heat_index_est"] = df["temp_avg"] + 0.1 * df["humidity_avg"]
    else:
        df["heat_index_est"] = None

    if "shortwave_radiation_sum" in df.columns and "sunshine_duration" in df.columns:
        df["solar_energy"] = (df["shortwave_radiation_sum"].fillna(0) *
                              df["sunshine_duration"].fillna(0) / 3600.0)
    else:
        df["solar_energy"] = None

    fill_map = {}
    for col in ["precipitation", "sunshine_duration", "windspeed_10m_max",
                "shortwave_radiation_sum", "et0_fao_evapotranspiration",
                "humidity_avg", "dew_point_avg", "cloud_cover_mean",
                "surface_pressure", "solar_energy", "heat_index_est", "temp_range"]:
        if col in df.columns:
            fill_map[col] = df[col].mean() if df[col].notna().any() else 0

    if fill_map:
        df.fillna(fill_map, inplace=True)

    round_cols = [c for c in ["temp_max", "temp_min", "temp_avg", "precipitation", "solar_energy"] if c in df.columns]
    for c in round_cols:
        df[c] = df[c].round(2)

    return df

def save_processed_data(df, city_name):
    os.makedirs("data/processed", exist_ok=True)
    file_path = f"data/processed/{city_name.lower()}_weather_clean.csv"
    df.to_csv(file_path, index=False)
    print(f"datos limpios guardados en {file_path}")