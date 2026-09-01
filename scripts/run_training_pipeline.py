#!/usr/bin/env python3
"""
CLI entry point for the training pipeline.

    python scripts/run_training_pipeline.py

Runs daily in production via `.github/workflows/training_pipeline.yml`.
Trains + evaluates every candidate model per forecast horizon and
registers the best one. Requires the feature store to already have data
(run `run_backfill.py` first).
"""

from __future__ import annotations

import argparse
import sys

from aqi_predictor.config import CITY_BY_KEY, DEFAULT_CITIES
from aqi_predictor.pipelines.training_pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register AQI forecasting models.")
    parser.add_argument("--cities", type=str, default=None,
                         help="Comma-separated city keys. Defaults to every configured city.")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                         help="Fraction of (chronologically most recent) dates held out "
                              "for evaluation (default: 0.2).")
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

    try:
        results = run_training_pipeline(cities=cities, test_fraction=args.test_fraction)
    except RuntimeError as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    print("\nTraining summary:")
    for horizon, result in sorted(results.items()):
        m = result.best.metrics
        print(f"  day+{horizon}: best={result.best.name:<14} "
              f"RMSE={m.rmse:.2f}  MAE={m.mae:.2f}  R2={m.r2:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
