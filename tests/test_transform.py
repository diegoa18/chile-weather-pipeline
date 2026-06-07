import numpy as np
import pandas as pd
import pytest

from src.etl.transform import clean_and_transform


def test_clean_and_transform_returns_expected_columns(sample_weather_df):
    df = clean_and_transform(sample_weather_df)
    expected = {"date", "temp_avg", "temp_range", "humidity_avg", "dew_point_avg"}
    assert expected.issubset(set(df.columns))


def test_clean_and_transform_no_nulls_in_critical(sample_weather_df):
    df = clean_and_transform(sample_weather_df)
    assert df["temp_avg"].notna().all()
    assert df["temp_max"].notna().all()
    assert df["temp_min"].notna().all()


def test_clean_and_transform_date_range(sample_weather_df):
    df = clean_and_transform(sample_weather_df)
    assert df["date"].min() == pd.Timestamp("2025-01-01")
    assert df["date"].max() == pd.Timestamp("2025-03-01")


def test_clean_and_transform_temp_avg_is_correct(sample_weather_df):
    raw_avg = (
        (sample_weather_df["temp_max"] + sample_weather_df["temp_min"]) / 2
    ).round(2)
    df = clean_and_transform(sample_weather_df)
    pd.testing.assert_series_equal(df["temp_avg"], raw_avg, check_names=False)


def test_clean_and_transform_drops_empty_date(sample_weather_df):
    sample_weather_df.loc[5, "date"] = None
    df = clean_and_transform(sample_weather_df)
    core_cols = ["temp_max", "temp_min", "temp_avg"]
    assert df[core_cols].isna().sum().sum() == 0


def test_clean_and_transform_rounding(sample_weather_df):
    df = clean_and_transform(sample_weather_df)
    for col in ["temp_max", "temp_min", "temp_avg", "precipitation"]:
        assert (np.round(df[col] * 100) % 1 == 0).all(), (
            f"{col} not rounded to 2 decimals"
        )


def test_clean_and_transform_empty_df():
    with pytest.raises((ValueError, KeyError)):
        clean_and_transform(pd.DataFrame())
