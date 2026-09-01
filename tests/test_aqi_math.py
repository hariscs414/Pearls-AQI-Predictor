"""Unit tests for aqi_predictor.data.aqi_math."""

from __future__ import annotations

import pytest

from aqi_predictor.data.aqi_math import ugm3_to_ppb, ugm3_to_ppm, us_aqi_from_components


def test_pm2_5_good_band():
    # 0-9.0 ug/m3 -> AQI 0-50 (per EPA 2024 breakpoints)
    aqi = us_aqi_from_components(pm2_5=4.5)
    assert 0 <= aqi <= 50


def test_pm2_5_moderate_band():
    # 9.1-35.4 ug/m3 -> AQI 51-100
    aqi = us_aqi_from_components(pm2_5=20.0)
    assert 51 <= aqi <= 100


def test_pm2_5_unhealthy_band():
    # 55.5-125.4 ug/m3 -> AQI 151-200
    aqi = us_aqi_from_components(pm2_5=90.0)
    assert 151 <= aqi <= 200


def test_consolidated_aqi_is_the_max_sub_index():
    # A very high PM2.5 alongside a low PM10 should be dominated by PM2.5.
    aqi_pm25_only = us_aqi_from_components(pm2_5=90.0)
    aqi_combined = us_aqi_from_components(pm2_5=90.0, pm10=10.0)
    assert aqi_combined == aqi_pm25_only


def test_none_inputs_return_none():
    assert us_aqi_from_components() is None


def test_result_is_monotonic_in_concentration():
    low = us_aqi_from_components(pm2_5=10.0)
    high = us_aqi_from_components(pm2_5=100.0)
    assert high > low


def test_extreme_value_above_table_clamped_not_none():
    aqi = us_aqi_from_components(pm2_5=1000.0)
    assert aqi is not None
    assert aqi > 400


@pytest.mark.parametrize("pollutant,mw", [("co", 28.01), ("no2", 46.0055), ("o3", 48.00)])
def test_ugm3_to_ppb_and_back_is_consistent(pollutant, mw):
    ppb = ugm3_to_ppb(100.0, pollutant)
    assert ppb > 0
    ppm = ugm3_to_ppm(100.0, pollutant)
    assert ppm == pytest.approx(ppb / 1000.0)
