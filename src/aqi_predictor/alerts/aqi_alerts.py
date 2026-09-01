"""
Hazardous-AQI alerting.

Scans forecasts for any day where predicted AQI reaches "Unhealthy" or
worse (`config.HAZARD_ALERT_THRESHOLD`, AQI >= 151) and produces `Alert`
objects the Streamlit app renders as banners. If `SLACK_WEBHOOK_URL` is
configured, the same alerts are also posted to Slack -- used by the daily
training-pipeline GitHub Action so hazardous forecasts reach people even
when nobody has the dashboard open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import requests

from aqi_predictor.config import HAZARD_ALERT_THRESHOLD, SLACK_WEBHOOK_URL, AQICategory
from aqi_predictor.models.forecaster import CityForecast
from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Alert:
    city_key: str
    city_name: str
    target_date: date
    horizon_days: int
    predicted_aqi: float
    category: AQICategory

    @property
    def message(self) -> str:
        return (
            f"{self.city_name}: AQI forecast to reach {self.predicted_aqi:.0f} "
            f"({self.category.name}) on {self.target_date:%a, %b %d} "
            f"({self.horizon_days} day(s) ahead). {self.category.health_message}"
        )


def check_forecast_for_alerts(forecast: CityForecast) -> list[Alert]:
    """Return one `Alert` per forecast day that reaches the hazard threshold."""
    alerts = []
    for point in forecast.points:
        if point.predicted_aqi >= HAZARD_ALERT_THRESHOLD:
            alerts.append(
                Alert(
                    city_key=forecast.city_key,
                    city_name=forecast.city_name,
                    target_date=point.target_date,
                    horizon_days=point.horizon_days,
                    predicted_aqi=point.predicted_aqi,
                    category=point.category,
                )
            )
    return alerts


def check_many_forecasts(forecasts: dict[str, CityForecast]) -> list[Alert]:
    """Convenience wrapper over `forecast_many()`'s output; ignores per-city errors."""
    alerts: list[Alert] = []
    for result in forecasts.values():
        if isinstance(result, CityForecast):
            alerts.extend(check_forecast_for_alerts(result))
    return alerts


def send_slack_alert(alert: Alert, webhook_url: str | None = None) -> bool:
    """POST one alert to Slack. Returns False (and logs a warning) on any failure."""
    url = webhook_url or SLACK_WEBHOOK_URL
    if not url:
        return False
    try:
        response = requests.post(url, json={"text": f":warning: {alert.message}"}, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException:
        logger.warning("Failed to send Slack alert for %s", alert.city_name, exc_info=True)
        return False


def notify_alerts(alerts: list[Alert]) -> None:
    """Log every alert, and forward to Slack if `SLACK_WEBHOOK_URL` is configured."""
    for alert in alerts:
        logger.warning("HAZARD ALERT: %s", alert.message)
        if SLACK_WEBHOOK_URL:
            send_slack_alert(alert)
