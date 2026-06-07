import json
import logging

import numpy as np
import pandas as pd

from src.utils.paths import city_slug, get_city_path
from src.utils.serializer import NumpyEncoder

logger = logging.getLogger(__name__)


def stationarity_test(series: pd.Series) -> dict:
    from statsmodels.tsa.stattools import adfuller

    result = {}
    try:
        adf_stat, p_value, used_lag, nobs, crit_values, _ = adfuller(
            series.dropna(), autolag="AIC"
        )
        result = {
            "test": "ADF",
            "statistic": round(adf_stat, 4),
            "p_value": round(p_value, 6),
            "is_stationary": p_value < 0.05,
            "critical_values": {k: round(v, 4) for k, v in crit_values.items()},
        }
    except Exception as e:
        logger.warning("ADF test fallo: %s", e)
        result = {"test": "ADF", "error": str(e)}
    return result


def seasonality_decomposition(df: pd.DataFrame, column: str = "temp_avg") -> dict:
    try:
        from statsmodels.tsa.seasonal import STL

        series = df.set_index("date")[column].dropna()
        if len(series) < 14:
            return {"error": f"serie muy corta ({len(series)} obs); minimo 14"}

        period = min(7, len(series) // 2)
        stl = STL(series, period=period).fit()

        trend_strength = 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid)
        seasonal_strength = 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid)

        return {
            "period": period,
            "trend_strength": round(float(trend_strength), 4),
            "seasonal_strength": round(float(seasonal_strength), 4),
            "trend_range": {
                "min": round(float(stl.trend.min()), 2),
                "max": round(float(stl.trend.max()), 2),
            },
            "seasonal_range": {
                "min": round(float(stl.seasonal.min()), 2),
                "max": round(float(stl.seasonal.max()), 2),
            },
            "residual_std": round(float(stl.resid.std()), 4),
        }
    except ImportError:
        return {"error": "statsmodels no instalado"}
    except Exception as e:
        logger.warning("STL descomposicion fallo en: %s", e)
        return {"error": str(e)}


def residual_analysis(y_true: pd.Series, y_pred: pd.Series) -> dict:
    residuals = np.array(y_true) - np.array(y_pred)

    from scipy import stats

    # Ljung-Box test para autocorrelación en residuos
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(residuals, lags=[7], return_df=True)
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])
    except ImportError:
        lb_pvalue = None

    # normalidad (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)

    return {
        "mean": round(float(np.mean(residuals)), 4),
        "std": round(float(np.std(residuals)), 4),
        "min": round(float(np.min(residuals)), 4),
        "max": round(float(np.max(residuals)), 4),
        "jarque_bera": {"statistic": round(jb_stat, 4), "p_value": round(jb_pvalue, 6)},
        "ljung_box_lag7": {"p_value": lb_pvalue} if lb_pvalue is not None else None,
        "is_normal": jb_pvalue > 0.05,
    }


def correlation_analysis(df: pd.DataFrame) -> dict:
    numeric_cols = [
        "temp_avg",
        "temp_max",
        "temp_min",
        "temp_range",
        "precipitation",
        "humidity_avg",
        "dew_point_avg",
        "windspeed_10m_max",
        "shortwave_radiation_sum",
        "cloud_cover_mean",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    if len(available) < 2:
        return {"error": "no hay suficientes columnas numericas"}

    corr = df[available].corr().round(3)

    # extraer correlaciones con temp_avg
    target_corr = {}
    if "temp_avg" in corr.columns:
        target_corr = corr["temp_avg"].drop("temp_avg").to_dict()

    return {
        "target_correlations": target_corr,
        "shape": list(corr.shape),
    }


def run_full_eda(df: pd.DataFrame, city_name: str) -> dict:
    eda_result = {"city": city_name}

    # estadísticas descriptivas extendidas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    desc = df[numeric_cols].describe().round(2)
    eda_result["descriptive_stats"] = desc.to_dict()

    # estacionariedad
    if "temp_avg" in df.columns:
        eda_result["stationarity"] = stationarity_test(df["temp_avg"])

    # estacionalidad
    eda_result["seasonality"] = seasonality_decomposition(df)

    # correlaciones
    eda_result["correlations"] = correlation_analysis(df)

    # valores faltantes
    missing = df.isna().sum()
    eda_result["missing_values"] = missing[missing > 0].to_dict()

    return eda_result


def save_eda_report(eda_result: dict, city_name: str) -> str:
    folder = get_city_path(city_name, "results")
    path = folder / f"{city_slug(city_name)}_eda.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eda_result, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    logger.info("reporte EDA guardado en: %s", path)
    return str(path)
