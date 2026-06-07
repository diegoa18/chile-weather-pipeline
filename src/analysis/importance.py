import json
import logging

import numpy as np
import pandas as pd

from src.utils.paths import city_slug, get_city_path
from src.utils.serializer import NumpyEncoder

logger = logging.getLogger(__name__)


def compute_shap_importance(model, X: pd.DataFrame, n_samples: int = 50) -> dict:
    try:
        import shap
    except ImportError:
        return {"error": "shap no instalado"}

    try:
        X_sample = X.sample(min(n_samples, len(X)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance = {}
        for i, col in enumerate(X_sample.columns):
            importance[col] = round(float(mean_abs_shap[i]), 4)

        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {
            "method": "SHAP TreeExplainer",
            "n_samples": n_samples,
            "importance": importance,
        }
    except Exception as e:
        logger.warning("SHAP falló: %s", e)
        return {"error": str(e)}


def compute_permutation_importance(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5
) -> dict:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
    )

    importance = {}
    for i, col in enumerate(X.columns):
        importance[col] = {
            "mean": round(float(result.importances_mean[i]), 4),
            "std": round(float(result.importances_std[i]), 4),
        }

    importance = dict(
        sorted(importance.items(), key=lambda x: x[1]["mean"], reverse=True)
    )
    return {"method": "permutation", "n_repeats": n_repeats, "importance": importance}


def run_feature_importance(model, X: pd.DataFrame, y: pd.Series) -> dict:
    logger.info("computando feature importance (SHAP + permutation)...")
    return {
        "shap": compute_shap_importance(model, X),
        "permutation": compute_permutation_importance(model, X, y),
    }


def save_importance_report(report: dict, city_name: str) -> str:
    folder = get_city_path(city_name, "results")
    path = folder / f"{city_slug(city_name)}_importance.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    logger.info("feature importance guardado en: %s", path)
    return str(path)
