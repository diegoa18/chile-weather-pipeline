import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_weather_df() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    np.random.seed(42)

    df = pd.DataFrame({"date": dates})
    df["temp_max"] = (
        20 + 5 * np.sin(np.linspace(0, 4 * np.pi, 60)) + np.random.normal(0, 1, 60)
    )
    df["temp_min"] = df["temp_max"] - np.random.uniform(5, 12, 60)
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["precipitation"] = np.random.exponential(2, 60)
    df.loc[:9, "precipitation"] = 0
    df["humidity_avg"] = np.random.uniform(40, 90, 60)
    df["dew_point_avg"] = np.random.uniform(5, 18, 60)
    df["windspeed_10m_max"] = np.random.uniform(5, 40, 60)
    df["shortwave_radiation_sum"] = np.random.uniform(10, 35, 60)
    df["et0_fao_evapotranspiration"] = np.random.uniform(1, 6, 60)
    df["cloud_cover_mean"] = np.random.uniform(20, 100, 60)
    df["sunshine_duration"] = np.random.uniform(100, 600, 60)
    df["weather_code"] = np.random.choice([0, 1, 2, 3, 45, 61, 80], 60)
    df["latitude"] = -33.45
    df["longitude"] = -70.66

    return df


@pytest.fixture
def sample_clean_df(sample_weather_df) -> pd.DataFrame:
    from src.etl.transform import clean_and_transform

    return clean_and_transform(sample_weather_df)
