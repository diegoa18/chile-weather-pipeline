import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.modeling.features.temporal_features import add_temporal_features
from src.modeling.train import _make_model, evaluate_model, prepare_training_data


def test_prepare_training_data_returns_xy(sample_clean_df):
    X, y, features = prepare_training_data(sample_clean_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(features) > 10
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(features)


def test_prepare_training_data_no_nulls(sample_clean_df):
    X, y, features = prepare_training_data(sample_clean_df)
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0


def test_make_model_returns_regressor(sample_clean_df):
    X, y, _ = prepare_training_data(sample_clean_df)
    model = _make_model()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert isinstance(preds[0], (np.floating, float))


def test_evaluate_model_returns_metrics(sample_clean_df):
    X, y, _ = prepare_training_data(sample_clean_df)
    metrics, y_true, y_pred = evaluate_model(X, y)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAE_std" in metrics
    assert "RMSE_std" in metrics
    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
    assert len(y_true) == len(y_pred)


def test_prepare_training_data_empty_df():
    with pytest.raises(ValueError, match="vacio"):
        prepare_training_data(pd.DataFrame())
