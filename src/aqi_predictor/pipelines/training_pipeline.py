"""
Training pipeline (runs daily in production, per the brief):

    1. Fetch historical (features, targets) from the Feature Store.
    2. Train and evaluate several candidate models per forecast horizon.
    3. Register the best model per horizon in the Model Registry.

Also computes and persists a SHAP global feature-importance table per
horizon (best-effort; a failure here never fails the whole pipeline) and,
once fresh models are registered, runs one forecast + hazard-alert check so
alerts fire the same day a new model goes live.
"""

from __future__ import annotations

import pandas as pd

from aqi_predictor.alerts.aqi_alerts import check_many_forecasts, notify_alerts
from aqi_predictor.config import DEFAULT_CITIES, FORECAST_HORIZON_DAYS, MIN_TRAINING_DAYS, City
from aqi_predictor.explainability.shap_explainer import global_feature_importance
from aqi_predictor.features.engineering import FEATURE_COLUMNS, training_frame
from aqi_predictor.features.feature_store import get_feature_store
from aqi_predictor.models.forecaster import forecast_many
from aqi_predictor.models.registry import get_model_registry
from aqi_predictor.models.trainer import TrainingResult, time_based_split, train_and_select_best
from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

MIN_TRAINING_ROWS = MIN_TRAINING_DAYS  # minimum usable rows (post-dropna) to attempt training


def _save_feature_importance(result: TrainingResult, train_df: pd.DataFrame,
                              test_df: pd.DataFrame, registry) -> None:
    try:
        importance = global_feature_importance(
            model=result.best.model,
            model_type=result.best.name,
            feature_columns=FEATURE_COLUMNS,
            background_df=train_df,
            sample_df=test_df,
        )
        out_path = registry.directory / f"feature_importance_h{result.horizon}d.csv"
        importance.to_csv(out_path, index=False)
        logger.info("Saved SHAP feature importance for horizon=%dd to %s",
                    result.horizon, out_path)
    except Exception:
        logger.exception(
            "Could not compute SHAP feature importance for horizon=%dd (non-fatal, "
            "the dashboard will fall back to computing it on demand).",
            result.horizon,
        )


def run_training_pipeline(
    cities: list[City] | None = None,
    test_fraction: float = 0.2,
    min_rows: int = MIN_TRAINING_ROWS,
    run_post_training_alert_check: bool = True,
) -> dict[int, TrainingResult]:
    cities = cities or DEFAULT_CITIES
    city_keys = [c.key for c in cities]

    store = get_feature_store()
    registry = get_model_registry()

    feature_df = store.read_features(city_keys=city_keys)
    if feature_df.empty:
        raise RuntimeError(
            "No features found in the feature store. Run the feature pipeline and/or "
            "backfill pipeline first: python scripts/run_backfill.py"
        )

    logger.info("Loaded %d feature rows for %d cities from the feature store.",
                len(feature_df), len(city_keys))

    results: dict[int, TrainingResult] = {}
    for horizon in range(1, FORECAST_HORIZON_DAYS + 1):
        target_col = f"target_{horizon}d"
        train_ready = training_frame(feature_df, FEATURE_COLUMNS, [target_col])

        if len(train_ready) < min_rows:
            logger.warning(
                "Only %d usable rows for horizon=%dd (need >= %d); skipping this horizon. "
                "Run the backfill pipeline with a longer lookback_days.",
                len(train_ready), horizon, min_rows,
            )
            continue

        try:
            train_df, test_df = time_based_split(train_ready, test_fraction=test_fraction)
        except ValueError as exc:
            logger.warning("Skipping horizon=%dd: %s", horizon, exc)
            continue

        if train_df.empty or test_df.empty:
            logger.warning(
                "Time-based split left an empty train or test set for horizon=%dd; skipping.",
                horizon,
            )
            continue

        result = train_and_select_best(train_df, test_df, FEATURE_COLUMNS, target_col, horizon)
        registry.save_model(
            model=result.best.model,
            horizon=horizon,
            model_type=result.best.name,
            metrics=result.best.metrics.as_dict(),
            feature_columns=FEATURE_COLUMNS,
        )
        _save_feature_importance(result, train_df, test_df, registry)
        results[horizon] = result

    if not results:
        raise RuntimeError(
            "Training pipeline produced no trained models: every horizon had too little "
            "data. Run the backfill pipeline with a longer lookback_days first."
        )

    if run_post_training_alert_check:
        try:
            forecasts = forecast_many(city_keys, store, registry)
            alerts = check_many_forecasts(forecasts)
            if alerts:
                notify_alerts(alerts)
            else:
                logger.info("Post-training forecast check: no hazardous AQI days ahead.")
        except Exception:
            logger.exception(
                "Post-training forecast/alert check failed (non-fatal; models are "
                "already registered)."
            )

    logger.info("Training pipeline complete. Trained horizons: %s", sorted(results.keys()))
    return results


if __name__ == "__main__":
    run_training_pipeline()
