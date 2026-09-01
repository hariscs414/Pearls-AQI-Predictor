"""
Feature engineering: turns raw hourly (weather + pollutant) rows into the
daily, model-ready feature table used for both training and inference.

Pipeline (see `build_feature_dataset` for the orchestrating function):

    raw hourly rows
        -> aggregate_hourly_to_daily      (1 row per city per day)
        -> add_time_features              (calendar + cyclical features)
        -> add_lag_and_rolling_features   (lags, rolling stats, AQI change rate)
        -> add_targets                    (AQI value 1/2/3 days ahead)

All grouped operations are done per `city_key` (via `groupby`) so cities
never leak information into each other's lag/rolling/target columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqi_predictor.config import FORECAST_HORIZON_DAYS

# ---------------------------------------------------------------------------
# Column name constants -- shared by the training pipeline, the forecaster
# and the Streamlit app so everyone agrees on the schema.
# ---------------------------------------------------------------------------
DAILY_AGG_COLUMNS = {
    "us_aqi": ["mean", "max", "min"],
    "pm2_5": ["mean"],
    "pm10": ["mean"],
    "carbon_monoxide": ["mean"],
    "nitrogen_dioxide": ["mean"],
    "sulphur_dioxide": ["mean"],
    "ozone": ["mean"],
    "dust": ["mean"],
    "temperature_2m": ["mean", "max", "min"],
    "relative_humidity_2m": ["mean"],
    "dew_point_2m": ["mean"],
    "apparent_temperature": ["mean"],
    "surface_pressure": ["mean"],
    "precipitation": ["sum"],
    "cloud_cover": ["mean"],
    "wind_speed_10m": ["mean", "max"],
    "wind_gusts_10m": ["max"],
}

LAG_DAYS = (1, 2, 3, 7)
ROLLING_WINDOWS = (3, 7)
BASE_TARGET = "us_aqi_mean"  # column the model predicts, N days ahead

TIME_FEATURES = [
    "day_of_week",
    "is_weekend",
    "month",
    "day_of_year",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

LAG_FEATURES = [f"us_aqi_mean_lag_{d}d" for d in LAG_DAYS]
ROLLING_FEATURES = [f"us_aqi_mean_rollmean_{w}d" for w in ROLLING_WINDOWS] + [
    f"us_aqi_mean_rollstd_{w}d" for w in ROLLING_WINDOWS
]
DERIVED_FEATURES = ["aqi_change_rate"] + LAG_FEATURES + ROLLING_FEATURES

WEATHER_FEATURES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "apparent_temperature_mean",
    "surface_pressure_mean",
    "precipitation_sum",
    "cloud_cover_mean",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

POLLUTANT_FEATURES = [
    "pm2_5_mean",
    "pm10_mean",
    "carbon_monoxide_mean",
    "nitrogen_dioxide_mean",
    "sulphur_dioxide_mean",
    "ozone_mean",
    "dust_mean",
]

FEATURE_COLUMNS = (
    ["us_aqi_mean", "us_aqi_max", "us_aqi_min"]
    + POLLUTANT_FEATURES
    + WEATHER_FEATURES
    + TIME_FEATURES
    + DERIVED_FEATURES
)

TARGET_COLUMNS = [f"target_{h}d" for h in range(1, FORECAST_HORIZON_DAYS + 1)]

ID_COLUMNS = ["city_key", "city_name", "date"]

# The exact columns produced directly by `aggregate_hourly_to_daily` (i.e.
# everything *before* time/lag/rolling/target features are added). The
# feature pipeline uses this to safely merge newly-fetched daily aggregates
# with previously-stored ones before recomputing derived features.
BASE_COLUMNS = ID_COLUMNS + [
    f"{col}_{stat}" for col, stats in DAILY_AGG_COLUMNS.items() for stat in stats
]


def aggregate_hourly_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly rows into one row per (city_key, date)."""
    if hourly_df.empty:
        return pd.DataFrame(columns=ID_COLUMNS)

    df = hourly_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date

    agg_map: dict[str, str] = {}
    rename_map: dict[str, str] = {}
    for col, stats in DAILY_AGG_COLUMNS.items():
        if col not in df.columns:
            continue
        for stat in stats:
            agg_key = f"{col}__{stat}"
            df[agg_key] = df[col]
            agg_map[agg_key] = stat
            rename_map[agg_key] = f"{col}_{stat}"

    grouped = df.groupby(["city_key", "city_name", "date"], as_index=False).agg(agg_map)
    grouped = grouped.rename(columns=rename_map)
    grouped = grouped.sort_values(["city_key", "date"]).reset_index(drop=True)
    return grouped


def add_time_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features (day of week, month, cyclical encodings)."""
    if daily_df.empty:
        return daily_df

    df = daily_df.copy()
    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_lag_and_rolling_features(
    daily_df: pd.DataFrame,
    target_col: str = BASE_TARGET,
    lags: tuple[int, ...] = LAG_DAYS,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Add per-city lag features, rolling mean/std, and the day-over-day AQI
    change rate. Every operation is grouped by `city_key` so no information
    leaks across cities, and rows are sorted by date first so lags/rolling
    windows are chronologically correct.
    """
    if daily_df.empty:
        return daily_df

    df = daily_df.sort_values(["city_key", "date"]).copy()
    grouped = df.groupby("city_key")[target_col]

    for lag in lags:
        df[f"{target_col}_lag_{lag}d"] = grouped.shift(lag)

    for window in windows:
        # shift(1) first so "today" is never included in its own rolling stat
        shifted = grouped.shift(1)
        df[f"{target_col}_rollmean_{window}d"] = shifted.groupby(df["city_key"]).transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        )
        df[f"{target_col}_rollstd_{window}d"] = shifted.groupby(df["city_key"]).transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).std()
        )

    lag1 = df[f"{target_col}_lag_1d"]
    df["aqi_change_rate"] = np.where(
        (lag1.notna()) & (lag1 != 0),
        (df[target_col] - lag1) / lag1,
        np.nan,
    )
    return df


def add_targets(
    daily_df: pd.DataFrame,
    target_col: str = BASE_TARGET,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    """Add `target_1d`...`target_Nd`: the AQI value N days *ahead*, per city."""
    if daily_df.empty:
        return daily_df

    df = daily_df.sort_values(["city_key", "date"]).copy()
    grouped = df.groupby("city_key")[target_col]
    for h in range(1, horizon_days + 1):
        df[f"target_{h}d"] = grouped.shift(-h)
    return df


def build_feature_dataset(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: raw hourly rows -> daily feature table with lags,
    rolling stats, calendar features and forward-looking targets.

    The returned frame still contains NaNs at the start (insufficient lag
    history) and end (insufficient future data for targets) of each city's
    series by design -- callers that need a clean training matrix should
    use `training_frame()` below, while the feature *store* should keep
    every row (NaNs and all) since the most recent rows are exactly what
    inference needs and naturally have empty target columns.
    """
    daily = aggregate_hourly_to_daily(hourly_df)
    daily = add_time_features(daily)
    daily = add_lag_and_rolling_features(daily)
    daily = add_targets(daily)
    return daily


def training_frame(
    feature_df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    target_columns: list[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Drop rows with missing features or targets, keeping only what's usable for training."""
    required = [c for c in (feature_columns + target_columns) if c in feature_df.columns]
    return feature_df.dropna(subset=required).reset_index(drop=True)


def latest_feature_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent row per city -- the input row used for live inference."""
    if feature_df.empty:
        return feature_df
    idx = feature_df.groupby("city_key")["date"].idxmax()
    return feature_df.loc[idx].reset_index(drop=True)
