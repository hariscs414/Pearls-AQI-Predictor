"""
Central configuration for the Pearls AQI Predictor.

Every tunable value used by more than one module lives here so behaviour can
be changed in one place. Values that differ between environments (API keys,
which backend to use, ...) are read from environment variables / a local
`.env` file via `python-dotenv`, with sensible defaults so the project runs
out of the box with zero configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` if present (no-op, and no error, if it is not).
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOCAL_STORE_DIR = DATA_DIR / "local_store"
FEATURE_STORE_DIR = LOCAL_STORE_DIR / "feature_store"
MODEL_REGISTRY_DIR = LOCAL_STORE_DIR / "model_registry"

for _dir in (DATA_DIR, LOCAL_STORE_DIR, FEATURE_STORE_DIR, MODEL_REGISTRY_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data provider
# ---------------------------------------------------------------------------
AQI_DATA_PROVIDER = os.getenv("AQI_DATA_PROVIDER", "open_meteo").strip().lower()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_MAX_PAST_DAYS = 92  # hard limit enforced by the Open-Meteo Air Quality API

OPENWEATHER_AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
OPENWEATHER_AIR_POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
OPENWEATHER_AIR_POLLUTION_FORECAST_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
OPENWEATHER_ONE_CALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

# ---------------------------------------------------------------------------
# Feature store / model registry backend
# ---------------------------------------------------------------------------
FEATURE_STORE_BACKEND = os.getenv("FEATURE_STORE_BACKEND", "local").strip().lower()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "").strip()
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip()

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_forecaster"

# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

# ---------------------------------------------------------------------------
# Forecast horizon
# ---------------------------------------------------------------------------
FORECAST_HORIZON_DAYS = 3  # predict AQI for day+1, day+2, day+3
MIN_TRAINING_DAYS = 30  # minimum number of daily rows required to train

# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class City:
    key: str
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str = "auto"


DEFAULT_CITIES: list[City] = [
    City("islamabad", "Islamabad", "Pakistan", 33.6844, 73.0479),
    City("lahore", "Lahore", "Pakistan", 31.5497, 74.3436),
    City("delhi", "Delhi", "India", 28.6139, 77.2090),
    City("beijing", "Beijing", "China", 39.9042, 116.4074),
    City("london", "London", "United Kingdom", 51.5072, -0.1276),
    City("los_angeles", "Los Angeles", "United States", 34.0522, -118.2437),
    City("sao_paulo", "Sao Paulo", "Brazil", -23.5505, -46.6333),
    City("lagos", "Lagos", "Nigeria", 6.5244, 3.3792),
]

CITY_BY_KEY: dict[str, City] = {c.key: c for c in DEFAULT_CITIES}

# ---------------------------------------------------------------------------
# Weather features fetched alongside pollutants (all instantaneous, hourly)
# ---------------------------------------------------------------------------
WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

AIR_QUALITY_HOURLY_VARS = [
    "us_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]

TARGET_COLUMN = "us_aqi"

# ---------------------------------------------------------------------------
# US EPA AQI categories (0-500 scale), used for the target variable, alerts
# and dashboard colour-coding.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AQICategory:
    name: str
    low: int
    high: int
    color: str
    health_message: str


AQI_CATEGORIES: list[AQICategory] = [
    AQICategory(
        "Good", 0, 50, "#00A65A",
        "Air quality is satisfactory and poses little or no risk.",
    ),
    AQICategory(
        "Moderate", 51, 100, "#E8C93A",
        "Air quality is acceptable. Unusually sensitive people should consider "
        "limiting prolonged outdoor exertion.",
    ),
    AQICategory(
        "Unhealthy for Sensitive Groups", 101, 150, "#F28C28",
        "Sensitive groups (children, elderly, people with respiratory or heart "
        "conditions) may experience health effects.",
    ),
    AQICategory(
        "Unhealthy", 151, 200, "#E0473E",
        "Everyone may begin to experience health effects; sensitive groups may "
        "experience more serious effects.",
    ),
    AQICategory(
        "Very Unhealthy", 201, 300, "#8B3A9E",
        "Health alert: everyone may experience more serious health effects. "
        "Avoid prolonged outdoor exertion.",
    ),
    AQICategory(
        "Hazardous", 301, 500, "#6E2430",
        "Health emergency: the entire population is more likely to be affected. "
        "Avoid all outdoor exertion.",
    ),
]

HAZARD_ALERT_THRESHOLD = 151  # AQI at/above this ("Unhealthy" and worse) triggers an alert


def categorize_aqi(aqi_value: float) -> AQICategory:
    """Return the AQICategory a given US AQI value falls into."""
    if aqi_value is None:
        raise ValueError("aqi_value must not be None")
    clamped = max(0, min(int(round(aqi_value)), 500))
    for category in AQI_CATEGORIES:
        if category.low <= clamped <= category.high:
            return category
    return AQI_CATEGORIES[-1]
