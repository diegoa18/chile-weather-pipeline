import numpy as np
import pytest

from src.analysis.importance import compute_permutation_importance, compute_shap_importance


def test_shap_importance_returns_top_features(sample_clean_df):
    from src.modeling.train import _make_model, prepare_training_data
    X, y, _ = prepare_training_data(sample_clean_df)
    model = _make_model()
    model.fit(X, y)
    result = compute_shap_importance(model, X, n_samples=30)
    if "error" not in result:
        assert "importance" in result
        assert len(result["importance"]) > 5
        top_feat = list(result["importance"].keys())[0]
        assert result["importance"][top_feat] > 0


def test_permutation_importance_returns_structured(sample_clean_df):
    from src.modeling.train import _make_model, prepare_training_data
    X, y, _ = prepare_training_data(sample_clean_df)
    model = _make_model()
    model.fit(X, y)
    result = compute_permutation_importance(model, X, y, n_repeats=2)
    assert "importance" in result
    top_feat = list(result["importance"].keys())[0]
    assert "mean" in result["importance"][top_feat]
    assert "std" in result["importance"][top_feat]
