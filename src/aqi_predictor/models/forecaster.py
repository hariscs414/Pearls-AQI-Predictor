"""
Turns the latest stored features + registered per-horizon models into an
actual N-day-ahead AQI forecast. This is what the Streamlit app calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from aqi_predictor.config import FORECAST_HORIZON_DAYS, AQICategory, categorize_aqi
from aqi_predictor.features.feature_store import FeatureStore
from aqi_predictor.models.registry import ModelRegistry
from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


class ForecastUnavailableError(RuntimeError):
    """Raised when a forecast can't be produced (missing features or model)."""


@dataclass
class ForecastPoint:
    target_date: date
    horizon_days: int
    predicted_aqi: float
    category: AQICategory


@dataclass
class CityForecast:
    city_key: str
    city_name: str
    as_of_date: date
    latest_observed_aqi: float
    points: list[ForecastPoint]


def _predict_one(model, meta: dict, feature_row: pd.Series) -> float:
    feature_columns = meta["feature_columns"]
    missing = [c for c in feature_columns if c not in feature_row.index]
    if missing:
        raise ForecastUnavailableError(
            f"Latest feature row is missing columns required by the model: {missing}"
        )
    X = feature_row[feature_columns].to_frame().T
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any(axis=None):
        raise ForecastUnavailableError(
            "Latest feature row contains NaNs in required columns "
            "(city likely needs more backfilled history for lag/rolling features)."
        )
    pred = float(model.predict(X)[0])
    return max(0.0, min(pred, 500.0))


def forecast_city(
    city_key: str,
    feature_store: FeatureStore,
    model_registry: ModelRegistry,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> CityForecast:
    """Produce a `CityForecast` for one city using the latest stored features."""
    latest = feature_store.read_latest([city_key])
    if latest.empty:
        raise ForecastUnavailableError(
            f"No features stored for city '{city_key}'. Run the feature pipeline "
            f"(and ideally the backfill pipeline) first."
        )
    row = latest.iloc[0]

    points: list[ForecastPoint] = []
    for h in range(1, horizon_days + 1):
        if not model_registry.has_model(h):
            raise ForecastUnavailableError(
                f"No trained model registered for horizon={h}d. Run "
                f"scripts/run_training_pipeline.py first."
            )
        model, meta = model_registry.load_model(h)
        predicted_aqi = _predict_one(model, meta, row)
        target_date = row["date"] + timedelta(days=h)
        points.append(
            ForecastPoint(
                target_date=target_date,
                horizon_days=h,
                predicted_aqi=predicted_aqi,
                category=categorize_aqi(predicted_aqi),
            )
        )

    return CityForecast(
        city_key=city_key,
        city_name=row.get("city_name", city_key),
        as_of_date=row["date"],
        latest_observed_aqi=float(row["us_aqi_mean"]),
        points=points,
    )


def forecast_many(
    city_keys: list[str],
    feature_store: FeatureStore,
    model_registry: ModelRegistry,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> dict[str, CityForecast | Exception]:
    """
    Forecast several cities at once. Returns a dict keyed by city_key whose
    value is either a `CityForecast` or the `Exception` raised for that city
    (so one city's missing data doesn't break the whole dashboard).
    """
    results: dict[str, CityForecast | Exception] = {}
    for city_key in city_keys:
        try:
            results[city_key] = forecast_city(
                city_key, feature_store, model_registry, horizon_days
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            logger.warning("Could not forecast city '%s': %s", city_key, exc)
            results[city_key] = exc
    return results
