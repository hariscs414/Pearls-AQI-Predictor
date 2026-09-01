"""
Historical backfill pipeline: runs the feature computation logic over a
range of past dates to build up enough training data for the training
pipeline (per the brief's "Backfill historical (features, targets)" step).

Requests are chunked (`chunk_days`) to keep each API call a reasonable
size and to checkpoint progress -- if city #4 of 8 fails partway through a
long backfill, cities #1-3 (and earlier chunks of #4) are already safely
stored.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

from aqi_predictor.config import DEFAULT_CITIES, City, MIN_TRAINING_DAYS
from aqi_predictor.data.api_client import get_client
from aqi_predictor.features.engineering import aggregate_hourly_to_daily
from aqi_predictor.features.feature_store import get_feature_store
from aqi_predictor.pipelines.feature_pipeline import merge_recompute_and_store
from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

_POLITE_DELAY_S = 0.3


def _date_chunks(start: date, end: date, chunk_days: int):
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def run_backfill_pipeline(
    cities: list[City] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    lookback_days: int = 180,
    chunk_days: int = 30,
) -> pd.DataFrame:
    """
    Backfill `[start_date, end_date]` (default: the last `lookback_days`
    days, ending yesterday) for `cities` (default: `config.DEFAULT_CITIES`).

    Returns the full recomputed feature table for the backfilled cities.
    """
    cities = cities or DEFAULT_CITIES
    client = get_client()
    store = get_feature_store()

    end_date = end_date or (date.today() - timedelta(days=1))
    start_date = start_date or (end_date - timedelta(days=lookback_days - 1))

    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    total_days = (end_date - start_date).days + 1
    logger.info(
        "Backfilling %d day(s) [%s .. %s] for %d city/cities via '%s' provider",
        total_days, start_date, end_date, len(cities), client.name,
    )

    all_daily_frames = []
    for city in cities:
        city_hourly_frames = []
        for chunk_start, chunk_end in _date_chunks(start_date, end_date, chunk_days):
            try:
                hourly = client.fetch_historical(city, chunk_start, chunk_end)
                if not hourly.empty:
                    city_hourly_frames.append(hourly)
                logger.info(
                    "  %-14s [%s .. %s]: %d hourly rows",
                    city.name, chunk_start, chunk_end, len(hourly),
                )
            except Exception:
                logger.exception(
                    "  %-14s [%s .. %s]: failed, skipping this chunk",
                    city.name, chunk_start, chunk_end,
                )
            time.sleep(_POLITE_DELAY_S)

        if not city_hourly_frames:
            logger.warning("No historical data retrieved for %s; skipping.", city.name)
            continue

        city_hourly = pd.concat(city_hourly_frames, ignore_index=True)
        all_daily_frames.append(aggregate_hourly_to_daily(city_hourly))

    if not all_daily_frames:
        logger.warning("Backfill produced no data for any city.")
        return pd.DataFrame()

    new_daily = pd.concat(all_daily_frames, ignore_index=True)
    city_keys = [c.key for c in cities]
    enriched = merge_recompute_and_store(new_daily, city_keys, store)

    n_days_per_city = enriched.groupby("city_key").size()
    thin_cities = n_days_per_city[n_days_per_city < MIN_TRAINING_DAYS]
    if not thin_cities.empty:
        logger.warning(
            "These cities have fewer than %d days of data and may not train well yet: %s",
            MIN_TRAINING_DAYS, dict(thin_cities),
        )

    logger.info("Backfill complete: %d total daily rows across %d cities.",
                len(enriched), len(city_keys))
    return enriched


if __name__ == "__main__":
    run_backfill_pipeline()
