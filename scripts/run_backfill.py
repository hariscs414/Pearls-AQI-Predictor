#!/usr/bin/env python3
"""
CLI entry point for the historical backfill pipeline.

    python scripts/run_backfill.py
    python scripts/run_backfill.py --cities islamabad --lookback-days 90
    python scripts/run_backfill.py --start-date 2025-01-01 --end-date 2025-06-01

Run this once before the first training run (and any time you add a new
city) to build up enough history for the lag/rolling features and targets.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from aqi_predictor.config import CITY_BY_KEY, DEFAULT_CITIES
from aqi_predictor.pipelines.backfill_pipeline import run_backfill_pipeline


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features.")
    parser.add_argument("--cities", type=str, default=None,
                         help="Comma-separated city keys. Defaults to every configured city.")
    parser.add_argument("--lookback-days", type=int, default=180,
                         help="Days of history to backfill, ending yesterday (default: 180). "
                              "Ignored if --start-date is given.")
    parser.add_argument("--start-date", type=_parse_date, default=None,
                         help="ISO date (YYYY-MM-DD) to start backfilling from.")
    parser.add_argument("--end-date", type=_parse_date, default=None,
                         help="ISO date (YYYY-MM-DD) to backfill through (default: yesterday).")
    parser.add_argument("--chunk-days", type=int, default=30,
                         help="Days per API request chunk (default: 30).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.cities:
        keys = [k.strip() for k in args.cities.split(",") if k.strip()]
        unknown = [k for k in keys if k not in CITY_BY_KEY]
        if unknown:
            print(f"Unknown city key(s): {unknown}. Known keys: {list(CITY_BY_KEY)}",
                  file=sys.stderr)
            return 1
        cities = [CITY_BY_KEY[k] for k in keys]
    else:
        cities = DEFAULT_CITIES

    df = run_backfill_pipeline(
        cities=cities,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        chunk_days=args.chunk_days,
    )
    print(f"Backfill finished: {len(df)} total daily rows for {len(cities)} city/cities.")
    return 0 if not df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
