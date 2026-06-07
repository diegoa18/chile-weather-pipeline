import pandas as pd
import pytest

from src.modeling.features.temporal_features import add_temporal_features


def test_add_temporal_features_lags(sample_weather_df):
    df, features = add_temporal_features(sample_weather_df)
    assert "temp_avg_lag_1" in df.columns
    assert "temp_avg_lag_3" in df.columns
    assert "temp_avg_lag_7" in df.columns
    assert "precipitation_lag_1" in df.columns


def test_add_temporal_features_rolling(sample_weather_df):
    df, features = add_temporal_features(sample_weather_df)
    assert "temp_avg_rolling_3" in df.columns
    assert "temp_avg_rolling_7" in df.columns


def test_add_temporal_features_cyclical(sample_weather_df):
    df, features = add_temporal_features(sample_weather_df)
    assert "day_of_year" in df.columns
    assert "month" in df.columns


def test_add_temporal_features_returns_feature_list(sample_weather_df):
    df, features = add_temporal_features(sample_weather_df)
    assert isinstance(features, list)
    assert len(features) > 0
    for f in features:
        assert f in df.columns


def test_add_temporal_features_no_date():
    with pytest.raises(ValueError, match="columna 'date' no encontrada"):
        add_temporal_features(pd.DataFrame({"a": [1, 2, 3]}))


def test_add_temporal_features_nans_at_start(sample_weather_df):
    df, _ = add_temporal_features(sample_weather_df)
    assert df["temp_avg_lag_7"].isna().sum() == 7
    assert df["temp_avg_lag_1"].isna().sum() == 1
