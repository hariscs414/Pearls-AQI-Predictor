"""Unit tests for aqi_predictor.config.categorize_aqi and aqi_predictor.alerts.aqi_alerts."""

from __future__ import annotations

from datetime import date

import pytest

from aqi_predictor.alerts.aqi_alerts import check_forecast_for_alerts, check_many_forecasts
from aqi_predictor.config import HAZARD_ALERT_THRESHOLD, categorize_aqi
from aqi_predictor.models.forecaster import CityForecast, ForecastPoint


@pytest.mark.parametrize(
    "aqi,expected_category",
    [
        (0, "Good"),
        (50, "Good"),
        (51, "Moderate"),
        (100, "Moderate"),
        (101, "Unhealthy for Sensitive Groups"),
        (150, "Unhealthy for Sensitive Groups"),
        (151, "Unhealthy"),
        (200, "Unhealthy"),
        (201, "Very Unhealthy"),
        (300, "Very Unhealthy"),
        (301, "Hazardous"),
        (500, "Hazardous"),
        (999, "Hazardous"),  # out-of-range clamps to the worst category
    ],
)
def test_categorize_aqi_boundaries(aqi, expected_category):
    assert categorize_aqi(aqi).name == expected_category


def _forecast_with(aqi_values: list[float]) -> CityForecast:
    points = [
        ForecastPoint(
            target_date=date.today(),
            horizon_days=i + 1,
            predicted_aqi=v,
            category=categorize_aqi(v),
        )
        for i, v in enumerate(aqi_values)
    ]
    return CityForecast(
        city_key="test", city_name="Test City", as_of_date=date.today(),
        latest_observed_aqi=aqi_values[0], points=points,
    )


def test_no_alert_below_threshold():
    forecast = _forecast_with([50, 80, 100])
    assert check_forecast_for_alerts(forecast) == []


def test_alert_fires_at_exact_threshold():
    forecast = _forecast_with([HAZARD_ALERT_THRESHOLD])
    alerts = check_forecast_for_alerts(forecast)
    assert len(alerts) == 1
    assert alerts[0].predicted_aqi == HAZARD_ALERT_THRESHOLD


def test_alert_fires_only_for_hazardous_days():
    forecast = _forecast_with([80, 160, 90])
    alerts = check_forecast_for_alerts(forecast)
    assert len(alerts) == 1
    assert alerts[0].horizon_days == 2


def test_check_many_forecasts_skips_errors():
    ok_forecast = _forecast_with([200])
    forecasts = {"good_city": _forecast_with([10]), "bad_city": RuntimeError("no data"),
                 "hazard_city": ok_forecast}
    alerts = check_many_forecasts(forecasts)
    assert len(alerts) == 1
    assert alerts[0].city_key == "test"  # from ok_forecast / hazard_city


def test_alert_message_contains_key_facts():
    forecast = _forecast_with([250])
    alert = check_forecast_for_alerts(forecast)[0]
    assert "Test City" in alert.message
    assert "250" in alert.message
    assert alert.category.name in alert.message
