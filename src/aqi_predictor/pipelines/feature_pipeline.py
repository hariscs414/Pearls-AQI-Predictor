"""
Feature pipeline (runs every hour in production, per the brief):

    1. Fetch raw weather + pollutant data from the configured provider.
    2. Compute daily features (time-based + derived, incl. AQI change rate).
    3. Store the result in the feature store.

Because the daily lag/rolling features depend on prior days, this pipeline
merges freshly-fetched raw aggregates with whatever is already in the
feature store for the same cities, recomputes every derived feature over
that union, and upserts the result. That keeps the logic simple and
correct (no partial-update bugs) at a dataset size where recomputation is
essentially free -- see `run_feature_pipeline`'s docstring for detail.
"""

from __future__ import annotations

import pandas as pd

from aqi_predictor.config import DEFAULT_CITIES, City, categorize_aqi, HAZARD_ALERT_THRESHOLD
from aqi_predictor.data.api_client import get_client
from aqi_predictor.features.engineering import (
    BASE_COLUMNS,
    add_lag_and_rolling_features,
    add_targets,
    add_time_features,
    aggregate_hourly_to_daily,
)
from aqi_predictor.features.feature_store import get_feature_store
from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_feature_pipeline(
    cities: list[City] | None = None,
    past_days: int = 3,
    forecast_days: int = 1,
) -> pd.DataFrame:
    """
    Fetch recent data for `cities`, recompute daily features, and upsert
    them into the feature store. Returns the full recomputed feature table
    for the given cities (useful for tests / notebooks).
    """
    cities = cities or DEFAULT_CITIES
    client = get_client()
    store = get_feature_store()

    logger.info(
        "Running feature pipeline for %d city/cities via '%s' provider "
        "(past_days=%d, forecast_days=%d)",
        len(cities), client.name, past_days, forecast_days,
    )

    hourly_frames = []
    for city in cities:
        try:
            hourly = client.fetch_recent(city, past_days=past_days, forecast_days=forecast_days)
        except Exception:
            logger.exception("Failed to fetch recent data for %s; skipping this city.", city.name)
            continue
        if hourly.empty:
            logger.warning("Provider returned no rows for %s; skipping.", city.name)
            continue
        hourly_frames.append(hourly)

    if not hourly_frames:
        logger.warning("No data fetched for any city; feature pipeline produced nothing.")
        return pd.DataFrame()

    hourly_df = pd.concat(hourly_frames, ignore_index=True)
    new_daily = aggregate_hourly_to_daily(hourly_df)
    city_keys = [c.key for c in cities]

    enriched = merge_recompute_and_store(new_daily, city_keys, store)
    _log_current_conditions(new_daily)

    logger.info("Feature pipeline complete: %d total daily rows for %d cities.",
                len(enriched), len(city_keys))
    return enriched


def merge_recompute_and_store(new_daily: pd.DataFrame, city_keys: list[str], store) -> pd.DataFrame:
    """
    Shared by both the hourly feature pipeline and the backfill pipeline:
    merge newly-aggregated daily rows with whatever the store already has
    for these cities, recompute every derived feature over the union, and
    upsert. Returns the full recomputed table for `city_keys`.
    """
    existing = store.read_features(city_keys=city_keys)
    if not existing.empty:
        existing_base = existing[[c for c in BASE_COLUMNS if c in existing.columns]]
        combined_base = pd.concat([existing_base, new_daily], ignore_index=True)
    else:
        combined_base = new_daily

    combined_base = (
        combined_base.drop_duplicates(subset=["city_key", "date"], keep="last")
        .sort_values(["city_key", "date"])
        .reset_index(drop=True)
    )

    enriched = add_time_features(combined_base)
    enriched = add_lag_and_rolling_features(enriched)
    enriched = add_targets(enriched)

    store.write_features(enriched)
    return enriched


def _log_current_conditions(new_daily: pd.DataFrame) -> None:
    """Best-effort warning log if today's observed AQI is already hazardous."""
    if new_daily.empty or "us_aqi_max" not in new_daily.columns:
        return
    latest_per_city = new_daily.loc[new_daily.groupby("city_key")["date"].idxmax()]
    for _, row in latest_per_city.iterrows():
        aqi_max = row.get("us_aqi_max")
        if pd.isna(aqi_max) or aqi_max < HAZARD_ALERT_THRESHOLD:
            continue
        category = categorize_aqi(aqi_max)
        logger.warning(
            "CURRENT CONDITIONS ALERT: %s observed AQI up to %.0f (%s) today.",
            row["city_name"], aqi_max, category.name,
        )


if __name__ == "__main__":
    run_feature_pipeline()
