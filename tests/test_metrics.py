from src.analysis.metrics import compute_weather_metrics


def test_compute_weather_metrics_structure(sample_clean_df):
    metrics = compute_weather_metrics(sample_clean_df)
    assert "date_range" in metrics
    assert "temperature" in metrics
    assert "precipitation" in metrics
    assert "humidity" in metrics
    assert "solar" in metrics
    assert "records_count" in metrics


def test_compute_weather_metrics_temperature_values(sample_clean_df):
    metrics = compute_weather_metrics(sample_clean_df)
    temp = metrics["temperature"]
    assert isinstance(temp["avg"], float)
    assert isinstance(temp["max_mean"], float)
    assert isinstance(temp["min_mean"], float)
    assert -10 < temp["avg"] < 50


def test_compute_weather_metrics_records_count(sample_clean_df):
    metrics = compute_weather_metrics(sample_clean_df)
    assert metrics["records_count"] == len(sample_clean_df)


def test_compute_weather_metrics_precipitation(sample_clean_df):
    metrics = compute_weather_metrics(sample_clean_df)
    assert metrics["precipitation"]["total"] >= 0
    assert metrics["precipitation"]["avg_daily"] >= 0


def test_compute_weather_metrics_empty_df():
    import pandas as pd
    df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                       "temp_avg": pd.Series(dtype="float64"),
                       "temp_max": pd.Series(dtype="float64"),
                       "temp_min": pd.Series(dtype="float64"),
                       "precipitation": pd.Series(dtype="float64")})
    metrics = compute_weather_metrics(df)
    assert metrics["records_count"] == 0
