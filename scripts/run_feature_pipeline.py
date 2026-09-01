#!/usr/bin/env python3
"""
CLI entry point for the feature pipeline.

    python scripts/run_feature_pipeline.py
    python scripts/run_feature_pipeline.py --cities islamabad,delhi --past-days 5

Runs every hour in production via `.github/workflows/feature_pipeline.yml`.
"""

from __future__ import annotations

import argparse
import sys

from aqi_predictor.config import CITY_BY_KEY, DEFAULT_CITIES
from aqi_predictor.pipelines.feature_pipeline import run_feature_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hourly AQI feature pipeline.")
    parser.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated city keys (see aqi_predictor.config.DEFAULT_CITIES). "
             "Defaults to every configured city.",
    )
    parser.add_argument("--past-days", type=int, default=3,
                         help="How many past days to refresh (default: 3).")
    parser.add_argument("--forecast-days", type=int, default=1,
                         help="How many days ahead to also fetch (default: 1).")
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

    df = run_feature_pipeline(cities=cities, past_days=args.past_days,
                               forecast_days=args.forecast_days)
    print(f"Feature pipeline finished: {len(df)} total daily rows for {len(cities)} city/cities.")
    return 0 if not df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
