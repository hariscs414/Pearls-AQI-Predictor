#!/usr/bin/env python3
"""
End-to-end smoke test / first-run demo.

    python scripts/run_full_demo.py
    python scripts/run_full_demo.py --cities islamabad --lookback-days 120

Runs backfill -> training -> forecast for a small set of cities and prints
a summary, so you can verify the whole system works right after cloning
the repo, before wiring up GitHub Actions or the dashboard.
"""

from __future__ import annotations

import argparse

from aqi_predictor.config import CITY_BY_KEY
from aqi_predictor.features.feature_store import get_feature_store
from aqi_predictor.models.forecaster import forecast_many
from aqi_predictor.models.registry import get_model_registry
from aqi_predictor.pipelines.backfill_pipeline import run_backfill_pipeline
from aqi_predictor.pipelines.training_pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full AQI pipeline end to end.")
    parser.add_argument("--cities", type=str, default="islamabad,delhi,london",
                         help="Comma-separated city keys (default: a small demo set).")
    parser.add_argument("--lookback-days", type=int, default=120,
                         help="Days of history to backfill (default: 120).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keys = [k.strip() for k in args.cities.split(",") if k.strip()]
    cities = [CITY_BY_KEY[k] for k in keys if k in CITY_BY_KEY]
    if not cities:
        print(f"No valid city keys in {keys!r}. Known keys: {list(CITY_BY_KEY)}")
        return 1

    print(f"=== 1/3 Backfilling {args.lookback_days} days for {[c.name for c in cities]} ===")
    run_backfill_pipeline(cities=cities, lookback_days=args.lookback_days)

    print("\n=== 2/3 Training models ===")
    run_training_pipeline(cities=cities)

    print("\n=== 3/3 Forecasting ===")
    store = get_feature_store()
    registry = get_model_registry()
    forecasts = forecast_many([c.key for c in cities], store, registry)

    for city in cities:
        result = forecasts[city.key]
        if isinstance(result, Exception):
            print(f"\n{city.name}: forecast unavailable ({result})")
            continue
        print(f"\n{city.name} (as of {result.as_of_date}, "
              f"latest observed AQI={result.latest_observed_aqi:.0f}):")
        for point in result.points:
            print(f"  day+{point.horizon_days} ({point.target_date}): "
                  f"AQI={point.predicted_aqi:.0f}  [{point.category.name}]")

    print("\nDone. Launch the dashboard with: streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
