"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_hourly_df() -> pd.DataFrame:
    """72 hours of synthetic hourly weather+AQI rows for a single city."""
    start = pd.Timestamp(date.today() - timedelta(days=3))
    idx = pd.date_range(start, periods=72, freq="h")
    rng = np.random.default_rng(0)

    df = pd.DataFrame({"datetime": idx})
    df["city_key"] = "testcity"
    df["city_name"] = "Test City"
    df["latitude"] = 10.0
    df["longitude"] = 20.0
    df["us_aqi"] = np.clip(80 + rng.normal(0, 10, len(idx)), 0, 500)
    df["pm10"] = np.clip(60 + rng.normal(0, 8, len(idx)), 0, 600)
    df["pm2_5"] = np.clip(40 + rng.normal(0, 6, len(idx)), 0, 500)
    df["carbon_monoxide"] = np.clip(300 + rng.normal(0, 20, len(idx)), 0, None)
    df["nitrogen_dioxide"] = np.clip(20 + rng.normal(0, 5, len(idx)), 0, None)
    df["sulphur_dioxide"] = np.clip(10 + rng.normal(0, 3, len(idx)), 0, None)
    df["ozone"] = np.clip(30 + rng.normal(0, 5, len(idx)), 0, None)
    df["dust"] = np.clip(5 + rng.normal(0, 2, len(idx)), 0, None)
    df["temperature_2m"] = 20 + rng.normal(0, 3, len(idx))
    df["relative_humidity_2m"] = np.clip(50 + rng.normal(0, 10, len(idx)), 0, 100)
    df["dew_point_2m"] = df["temperature_2m"] - 5
    df["apparent_temperature"] = df["temperature_2m"] + rng.normal(0, 1, len(idx))
    df["surface_pressure"] = 1013 + rng.normal(0, 4, len(idx))
    df["precipitation"] = np.clip(rng.exponential(0.02, len(idx)) - 0.015, 0, None)
    df["cloud_cover"] = np.clip(40 + rng.normal(0, 20, len(idx)), 0, 100)
    df["wind_speed_10m"] = np.clip(10 + rng.normal(0, 4, len(idx)), 0, None)
    df["wind_direction_10m"] = rng.uniform(0, 360, len(idx))
    df["wind_gusts_10m"] = df["wind_speed_10m"] + rng.normal(3, 1, len(idx))
    return df


@pytest.fixture
def multi_day_daily_df() -> pd.DataFrame:
    """20 days of a single city's already-aggregated daily rows (for lag/rolling tests)."""
    dates = [date.today() - timedelta(days=d) for d in range(20, 0, -1)]
    rng = np.random.default_rng(1)
    values = 100 + np.cumsum(rng.normal(0, 5, len(dates)))
    df = pd.DataFrame(
        {
            "city_key": "testcity",
            "city_name": "Test City",
            "date": dates,
            "us_aqi_mean": values,
        }
    )
    return df
