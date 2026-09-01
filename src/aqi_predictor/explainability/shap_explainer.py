"""
SHAP-based feature-importance explanations for the registered AQI models.

Tree-based models (`random_forest`, `xgboost`) use the fast, exact
`shap.TreeExplainer`. Every other model type (`ridge`, `dense_nn`) falls
back to the model-agnostic `shap.Explainer` driven off `model.predict`,
which works for any scikit-learn-compatible estimator at the cost of being
slower -- fine here since explanations are computed for a handful of rows
at a time (one per city), not the whole training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

_TREE_MODEL_TYPES = {"random_forest", "xgboost"}
_MAX_BACKGROUND_ROWS = 100


def _get_shap_values(model, model_type: str, background_df: pd.DataFrame,
                      explain_df: pd.DataFrame) -> np.ndarray:
    if model_type in _TREE_MODEL_TYPES:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(explain_df)
    else:
        background = background_df
        if len(background) > _MAX_BACKGROUND_ROWS:
            background = shap.sample(background, _MAX_BACKGROUND_ROWS, random_state=42)
        explainer = shap.Explainer(model.predict, background)
        values = explainer(explain_df).values
    return np.asarray(values)


def global_feature_importance(
    model,
    model_type: str,
    feature_columns: list[str],
    background_df: pd.DataFrame,
    sample_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Mean |SHAP value| per feature over `sample_df` (or `background_df` if no
    separate sample is given) -- a global "what drives predictions" ranking,
    used for the model-report chart and dashboard.
    """
    explain_df = sample_df if sample_df is not None else background_df
    if len(explain_df) > _MAX_BACKGROUND_ROWS:
        explain_df = explain_df.sample(_MAX_BACKGROUND_ROWS, random_state=42)

    X_background = background_df[feature_columns]
    X_explain = explain_df[feature_columns]

    values = _get_shap_values(model, model_type, X_background, X_explain)
    mean_abs = np.abs(values).mean(axis=0)

    importance = pd.DataFrame(
        {"feature": feature_columns, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance


def explain_single_prediction(
    model,
    model_type: str,
    feature_columns: list[str],
    background_df: pd.DataFrame,
    row: pd.Series,
    top_k: int = 8,
) -> pd.DataFrame:
    """
    Per-prediction explanation for one feature row (e.g. "why is tomorrow's
    forecast 142?"): the `top_k` features with the largest |SHAP value|,
    signed so the dashboard can show "pushed the forecast up/down".
    """
    X_background = background_df[feature_columns]
    X_row = row[feature_columns].to_frame().T
    # A single row sliced out of a DataFrame with mixed dtypes (plus
    # non-feature columns like city_key/date) collapses to an object-dtype
    # Series; coerce back to numeric so SHAP's masker gets a clean float
    # array instead of choking on dtype=object.
    X_row = X_row.apply(pd.to_numeric, errors="coerce")

    values = _get_shap_values(model, model_type, X_background, X_row)[0]
    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "shap_value": values,
            "feature_value": row[feature_columns].to_numpy(),
        }
    )
    result["abs_shap"] = result["shap_value"].abs()
    result = result.sort_values("abs_shap", ascending=False).head(top_k)
    return result.drop(columns="abs_shap").reset_index(drop=True)
