"""Unit tests for aqi_predictor.features.engineering."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from aqi_predictor.features.engineering import (
    add_lag_and_rolling_features,
    add_targets,
    add_time_features,
    aggregate_hourly_to_daily,
    latest_feature_rows,
    training_frame,
)


def test_aggregate_hourly_to_daily_one_row_per_city_per_day(sample_hourly_df):
    daily = aggregate_hourly_to_daily(sample_hourly_df)

    n_expected_days = sample_hourly_df["datetime"].dt.date.nunique()
    assert len(daily) == n_expected_days
    assert set(["city_key", "city_name", "date"]).issubset(daily.columns)
    assert "us_aqi_mean" in daily.columns
    assert "us_aqi_max" in daily.columns
    assert "us_aqi_min" in daily.columns
    # max >= mean >= min for every day
    assert (daily["us_aqi_max"] >= daily["us_aqi_mean"]).all()
    assert (daily["us_aqi_mean"] >= daily["us_aqi_min"]).all()


def test_aggregate_hourly_to_daily_empty_input_returns_empty():
    empty = pd.DataFrame(columns=["datetime", "city_key", "city_name"])
    result = aggregate_hourly_to_daily(empty)
    assert result.empty


def test_add_time_features_weekend_flag():
    df = pd.DataFrame({"date": [date(2024, 1, 6), date(2024, 1, 8)]})  # Sat, Mon
    out = add_time_features(df)
    assert out.loc[0, "is_weekend"] == 1
    assert out.loc[1, "is_weekend"] == 0
    assert out.loc[0, "day_of_week"] == 5  # Saturday


def test_add_time_features_cyclical_encoding_bounded():
    df = pd.DataFrame({"date": [date.today() - timedelta(days=i) for i in range(30)]})
    out = add_time_features(df)
    for col in ("month_sin", "month_cos", "doy_sin", "doy_cos"):
        assert out[col].between(-1.0001, 1.0001).all()


def test_lag_features_use_only_past_values(multi_day_daily_df):
    out = add_lag_and_rolling_features(multi_day_daily_df)
    out = out.sort_values("date").reset_index(drop=True)

    # lag_1d on row i should exactly equal us_aqi_mean on row i-1
    for i in range(1, len(out)):
        assert np.isclose(out.loc[i, "us_aqi_mean_lag_1d"], out.loc[i - 1, "us_aqi_mean"])

    # first row has no history -> lag columns are NaN
    assert pd.isna(out.loc[0, "us_aqi_mean_lag_1d"])


def test_rolling_mean_excludes_current_day(multi_day_daily_df):
    out = add_lag_and_rolling_features(multi_day_daily_df, windows=(3,))
    out = out.sort_values("date").reset_index(drop=True)

    # Manually compute rolling(3) mean of the *previous* 3 days for a mid-series row
    i = 10
    expected = multi_day_daily_df["us_aqi_mean"].iloc[i - 3:i].mean()
    assert np.isclose(out.loc[i, "us_aqi_mean_rollmean_3d"], expected)


def test_aqi_change_rate_matches_manual_calc(multi_day_daily_df):
    out = add_lag_and_rolling_features(multi_day_daily_df)
    out = out.sort_values("date").reset_index(drop=True)
    i = 5
    today = out.loc[i, "us_aqi_mean"]
    yesterday = out.loc[i - 1, "us_aqi_mean"]
    expected = (today - yesterday) / yesterday
    assert np.isclose(out.loc[i, "aqi_change_rate"], expected)


def test_lags_and_rolling_do_not_leak_across_cities():
    dates = [date.today() - timedelta(days=d) for d in range(10, 0, -1)]
    df = pd.concat(
        [
            pd.DataFrame({"city_key": "a", "city_name": "A", "date": dates,
                          "us_aqi_mean": [10.0] * len(dates)}),
            pd.DataFrame({"city_key": "b", "city_name": "B", "date": dates,
                          "us_aqi_mean": [999.0] * len(dates)}),
        ],
        ignore_index=True,
    )
    out = add_lag_and_rolling_features(df)
    city_a = out[out["city_key"] == "a"].sort_values("date")
    # every lag/rolling value for city 'a' must come from city 'a' (value 10),
    # never from city 'b' (value 999)
    assert (city_a["us_aqi_mean_lag_1d"].dropna() == 10.0).all()
    assert (city_a["us_aqi_mean_rollmean_3d"].dropna() == 10.0).all()


def test_add_targets_shifts_forward_per_city(multi_day_daily_df):
    out = add_targets(multi_day_daily_df, horizon_days=3)
    out = out.sort_values("date").reset_index(drop=True)
    for i in range(len(out) - 1):
        assert np.isclose(out.loc[i, "target_1d"], out.loc[i + 1, "us_aqi_mean"])
    # last row(s) have no future data -> NaN targets
    assert pd.isna(out.loc[len(out) - 1, "target_1d"])


def test_training_frame_drops_rows_with_missing_features_or_targets():
    df = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, np.nan, 4.0],
            "target_1d": [1.0, np.nan, 3.0, 4.0],
        }
    )
    out = training_frame(df, feature_columns=["feat_a"], target_columns=["target_1d"])
    assert len(out) == 2  # only rows 0 and 3 have both feat_a and target_1d present


def test_latest_feature_rows_picks_max_date_per_city():
    dates_a = [date.today() - timedelta(days=d) for d in (3, 2, 1)]
    dates_b = [date.today() - timedelta(days=d) for d in (5, 4)]
    df = pd.concat(
        [
            pd.DataFrame({"city_key": "a", "date": dates_a, "val": [1, 2, 3]}),
            pd.DataFrame({"city_key": "b", "date": dates_b, "val": [10, 20]}),
        ],
        ignore_index=True,
    )
    out = latest_feature_rows(df)
    assert len(out) == 2
    assert out.loc[out["city_key"] == "a", "val"].iloc[0] == 3
    assert out.loc[out["city_key"] == "b", "val"].iloc[0] == 20
