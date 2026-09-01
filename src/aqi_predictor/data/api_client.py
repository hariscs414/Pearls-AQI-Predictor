"""
Weather + Air Quality API clients.

Two adapters are provided behind a common `BaseAQIClient` interface:

- `OpenMeteoClient` (default): https://open-meteo.com -- free, no API key,
  generous rate limits, and returns a properly-computed US AQI (`us_aqi`)
  directly, so it is used as-is. This is the recommended provider and is
  what `get_client()` returns unless configured otherwise.

- `OpenWeatherClient`: https://openweathermap.org -- requires a free API
  key. Its Air Pollution API returns raw pollutant concentrations rather
  than a 0-500 US AQI, so `aqi_predictor.data.aqi_math.us_aqi_from_components`
  is used to derive a comparable target. Its free tier also has a shorter
  forecast horizon and no historical *weather* endpoint, which is called
  out explicitly where it matters.

Every method returns a tidy `pandas.DataFrame` with one row per
(datetime, city) and the following columns:

    datetime, city_key, city_name, latitude, longitude,
    <weather columns from config.WEATHER_HOURLY_VARS>,
    <air-quality columns from config.AIR_QUALITY_HOURLY_VARS, incl. us_aqi>

so the rest of the pipeline never needs to know which provider produced
the data.
"""

from __future__ import annotations

import abc
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aqi_predictor import config
from aqi_predictor.data.aqi_math import us_aqi_from_components
from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

_REQUEST_TIMEOUT_S = 30


class DataFetchError(RuntimeError):
    """Raised when a weather/AQI provider request fails after retries."""


def _retry_decorator():
    return retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.RequestException, DataFetchError)),
    )


@_retry_decorator()
def _get_json(url: str, params: dict) -> dict:
    """GET `url` with `params`, retrying on transient network/HTTP errors."""
    response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
    if response.status_code >= 500:
        raise DataFetchError(f"{url} returned server error {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise DataFetchError(f"{url} returned non-JSON response") from exc

    if response.status_code >= 400:
        reason = payload.get("message") or payload.get("reason") or payload
        raise DataFetchError(f"{url} returned {response.status_code}: {reason}")

    return payload


class BaseAQIClient(abc.ABC):
    """Common interface every weather/AQI provider adapter implements."""

    name: str = "base"

    @abc.abstractmethod
    def fetch_current(self, city: config.City) -> pd.DataFrame:
        """Return a single-row DataFrame with the latest available reading."""

    @abc.abstractmethod
    def fetch_forecast(self, city: config.City, forecast_days: int = 3) -> pd.DataFrame:
        """Return hourly forecast rows for the next `forecast_days` days."""

    @abc.abstractmethod
    def fetch_historical(
        self, city: config.City, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Return hourly historical rows for `[start_date, end_date]` inclusive."""

    def fetch_recent(
        self, city: config.City, past_days: int = 3, forecast_days: int = 1
    ) -> pd.DataFrame:
        """
        Return hourly rows spanning the last `past_days` days through
        `forecast_days` days ahead -- what the *hourly* feature pipeline
        uses to keep the last few days (including "today so far") fresh.

        Default implementation stitches `fetch_historical` + `fetch_forecast`
        together; `OpenMeteoClient` overrides this with a single, more
        efficient call per endpoint (and avoids the reanalysis-archive's
        multi-day publication delay -- see its docstring).
        """
        today = date.today()
        frames = [self.fetch_forecast(city, forecast_days=forecast_days)]
        if past_days > 0:
            start = today - timedelta(days=past_days)
            end = today - timedelta(days=1)
            frames.append(self.fetch_historical(city, start, end))
        combined = pd.concat(frames, ignore_index=True)
        return combined.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(
            drop=True
        )


def _attach_city_columns(df: pd.DataFrame, city: config.City) -> pd.DataFrame:
    df = df.copy()
    df["city_key"] = city.key
    df["city_name"] = city.name
    df["latitude"] = city.latitude
    df["longitude"] = city.longitude
    ordered = (
        ["datetime", "city_key", "city_name", "latitude", "longitude"]
        + [c for c in config.WEATHER_HOURLY_VARS if c in df.columns]
        + [c for c in config.AIR_QUALITY_HOURLY_VARS if c in df.columns]
    )
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


# ---------------------------------------------------------------------------
# Open-Meteo (default provider)
# ---------------------------------------------------------------------------
class OpenMeteoClient(BaseAQIClient):
    """Free, key-less weather + air-quality provider. See module docstring."""

    name = "open_meteo"

    @staticmethod
    def _hourly_to_frame(payload: dict, variables: list[str]) -> pd.DataFrame:
        hourly = payload.get("hourly")
        if not hourly or "time" not in hourly:
            return pd.DataFrame(columns=["datetime", *variables])
        data = {"datetime": pd.to_datetime(hourly["time"])}
        for var in variables:
            if var in hourly:
                data[var] = hourly[var]
        return pd.DataFrame(data)

    def _fetch_weather_hourly(
        self,
        city: config.City,
        *,
        historical: bool,
        start_date: date | None = None,
        end_date: date | None = None,
        forecast_days: int | None = None,
        past_days: int | None = None,
    ) -> pd.DataFrame:
        url = config.OPEN_METEO_ARCHIVE_URL if historical else config.OPEN_METEO_FORECAST_URL
        params = {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "hourly": ",".join(config.WEATHER_HOURLY_VARS),
            "timezone": city.timezone,
        }
        if historical:
            params["start_date"] = start_date.isoformat()
            params["end_date"] = end_date.isoformat()
        else:
            params["forecast_days"] = forecast_days
            if past_days is not None:
                params["past_days"] = past_days

        payload = _get_json(url, params)
        return self._hourly_to_frame(payload, config.WEATHER_HOURLY_VARS)

    def _fetch_air_quality_hourly(
        self,
        city: config.City,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        forecast_days: int | None = None,
        past_days: int | None = None,
    ) -> pd.DataFrame:
        params = {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "hourly": ",".join(config.AIR_QUALITY_HOURLY_VARS),
            "timezone": city.timezone,
        }
        if start_date is not None and end_date is not None:
            params["start_date"] = start_date.isoformat()
            params["end_date"] = end_date.isoformat()
        if forecast_days is not None:
            params["forecast_days"] = forecast_days
        if past_days is not None:
            params["past_days"] = past_days

        payload = _get_json(config.OPEN_METEO_AIR_QUALITY_URL, params)
        return self._hourly_to_frame(payload, config.AIR_QUALITY_HOURLY_VARS)

    def fetch_current(self, city: config.City) -> pd.DataFrame:
        weather_payload = _get_json(
            config.OPEN_METEO_FORECAST_URL,
            {
                "latitude": city.latitude,
                "longitude": city.longitude,
                "current": ",".join(config.WEATHER_HOURLY_VARS),
                "timezone": city.timezone,
            },
        )
        aq_payload = _get_json(
            config.OPEN_METEO_AIR_QUALITY_URL,
            {
                "latitude": city.latitude,
                "longitude": city.longitude,
                "current": ",".join(config.AIR_QUALITY_HOURLY_VARS),
                "timezone": city.timezone,
            },
        )

        weather_current = weather_payload.get("current", {})
        aq_current = aq_payload.get("current", {})
        ts = weather_current.get("time") or aq_current.get("time")
        if ts is None:
            raise DataFetchError(f"Open-Meteo returned no current reading for {city.name}")

        row = {"datetime": pd.to_datetime(ts)}
        for var in config.WEATHER_HOURLY_VARS:
            row[var] = weather_current.get(var)
        for var in config.AIR_QUALITY_HOURLY_VARS:
            row[var] = aq_current.get(var)

        df = pd.DataFrame([row])
        return _attach_city_columns(df, city)

    def fetch_forecast(self, city: config.City, forecast_days: int = 3) -> pd.DataFrame:
        # Air quality forecasts are capped at 7 days by the API; weather at 16,
        # so the binding constraint is air quality.
        forecast_days = max(1, min(forecast_days, 7))
        weather_df = self._fetch_weather_hourly(
            city, historical=False, forecast_days=forecast_days
        )
        aq_df = self._fetch_air_quality_hourly(city, forecast_days=forecast_days)
        merged = pd.merge(weather_df, aq_df, on="datetime", how="outer").sort_values("datetime")
        return _attach_city_columns(merged, city)

    def fetch_historical(
        self, city: config.City, start_date: date, end_date: date
    ) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")

        weather_df = self._fetch_weather_hourly(
            city, historical=True, start_date=start_date, end_date=end_date
        )
        try:
            aq_df = self._fetch_air_quality_hourly(
                city, start_date=start_date, end_date=end_date
            )
        except DataFetchError as exc:
            logger.warning(
                "Open-Meteo air-quality history unavailable for %s in [%s, %s] (%s). "
                "Global CAMS air-quality data is only available from 2022-08 onwards.",
                city.name,
                start_date,
                end_date,
                exc,
            )
            aq_df = pd.DataFrame(columns=["datetime", *config.AIR_QUALITY_HOURLY_VARS])

        merged = pd.merge(weather_df, aq_df, on="datetime", how="outer").sort_values("datetime")
        return _attach_city_columns(merged, city)

    def fetch_recent(
        self, city: config.City, past_days: int = 3, forecast_days: int = 1
    ) -> pd.DataFrame:
        """
        Recent + near-term data in one call per endpoint, via the *forecast*
        API's `past_days` parameter -- not the `/v1/archive` reanalysis
        endpoint, which is only updated with a ~5 day delay and would
        silently return nothing useful for "yesterday" or "today". This is
        what the hourly feature pipeline calls.
        """
        past_days = max(0, min(past_days, config.OPEN_METEO_MAX_PAST_DAYS))
        weather_df = self._fetch_weather_hourly(
            city,
            historical=False,
            forecast_days=forecast_days,
            past_days=past_days,
        )
        aq_df = self._fetch_air_quality_hourly(
            city, forecast_days=forecast_days, past_days=past_days
        )
        merged = pd.merge(weather_df, aq_df, on="datetime", how="outer").sort_values("datetime")
        return _attach_city_columns(merged, city)


# ---------------------------------------------------------------------------
# OpenWeather (optional alternate provider)
# ---------------------------------------------------------------------------
class OpenWeatherClient(BaseAQIClient):
    """
    Alternate provider using OpenWeather's free-tier endpoints.

    Notes / limitations (see module docstring for detail):
    - AQI is *derived* from raw pollutant concentrations via
      `aqi_math.us_aqi_from_components`, using instantaneous readings
      against EPA breakpoints (an approximation of the true rolling-average
      US AQI).
    - Air-pollution forecasts are limited to 4 days by OpenWeather.
    - There is no free historical *weather* endpoint; `fetch_historical`
      backfills pollutant/AQI history only and leaves weather columns NaN,
      logging a warning. Use `AQI_DATA_PROVIDER=open_meteo` for full
      historical backfill.
    """

    name = "openweather"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.OPENWEATHER_API_KEY
        if not self.api_key:
            raise ValueError(
                "OPENWEATHER_API_KEY is not set. Get a free key at "
                "https://openweathermap.org/api or set AQI_DATA_PROVIDER=open_meteo."
            )

    @staticmethod
    def _components_to_row(entry: dict) -> dict:
        components = entry.get("components", {})
        aqi = us_aqi_from_components(
            pm2_5=components.get("pm2_5"),
            pm10=components.get("pm10"),
            co=components.get("co"),
            no2=components.get("no2"),
            so2=components.get("so2"),
            o3=components.get("o3"),
            gases_in_ugm3=True,
        )
        return {
            "datetime": pd.to_datetime(entry["dt"], unit="s", utc=True).tz_localize(None),
            "us_aqi": aqi,
            "pm10": components.get("pm10"),
            "pm2_5": components.get("pm2_5"),
            "carbon_monoxide": components.get("co"),
            "nitrogen_dioxide": components.get("no2"),
            "sulphur_dioxide": components.get("so2"),
            "ozone": components.get("o3"),
            "dust": None,
        }

    @staticmethod
    def _weather_entry_to_row(entry: dict) -> dict:
        main = entry.get("main", {})
        wind = entry.get("wind", {})
        clouds = entry.get("clouds", {})
        return {
            "temperature_2m": main.get("temp"),
            "relative_humidity_2m": main.get("humidity"),
            "dew_point_2m": None,
            "apparent_temperature": main.get("feels_like"),
            "surface_pressure": main.get("pressure"),
            "precipitation": (entry.get("rain", {}) or {}).get("3h")
            or (entry.get("rain", {}) or {}).get("1h"),
            "cloud_cover": clouds.get("all"),
            "wind_speed_10m": wind.get("speed"),
            "wind_direction_10m": wind.get("deg"),
            "wind_gusts_10m": wind.get("gust"),
        }

    def fetch_current(self, city: config.City) -> pd.DataFrame:
        aq_payload = _get_json(
            config.OPENWEATHER_AIR_POLLUTION_URL,
            {"lat": city.latitude, "lon": city.longitude, "appid": self.api_key},
        )
        weather_payload = _get_json(
            "https://api.openweathermap.org/data/2.5/weather",
            {
                "lat": city.latitude,
                "lon": city.longitude,
                "appid": self.api_key,
                "units": "metric",
            },
        )

        aq_entries = aq_payload.get("list", [])
        if not aq_entries:
            raise DataFetchError(f"OpenWeather returned no air-pollution data for {city.name}")

        row = self._components_to_row(aq_entries[0])
        row.update(self._weather_entry_to_row(weather_payload))
        row["datetime"] = pd.to_datetime(
            weather_payload.get("dt"), unit="s", utc=True
        ).tz_localize(None)

        df = pd.DataFrame([row])
        return _attach_city_columns(df, city)

    def fetch_forecast(self, city: config.City, forecast_days: int = 3) -> pd.DataFrame:
        if forecast_days > 4:
            logger.warning(
                "OpenWeather's free Air Pollution forecast only covers 4 days; "
                "capping forecast_days from %s to 4.",
                forecast_days,
            )
            forecast_days = 4

        aq_payload = _get_json(
            config.OPENWEATHER_AIR_POLLUTION_FORECAST_URL,
            {"lat": city.latitude, "lon": city.longitude, "appid": self.api_key},
        )
        weather_payload = _get_json(
            "https://api.openweathermap.org/data/2.5/forecast",
            {
                "lat": city.latitude,
                "lon": city.longitude,
                "appid": self.api_key,
                "units": "metric",
            },
        )

        cutoff = pd.Timestamp.utcnow().tz_localize(None) + timedelta(days=forecast_days)

        aq_rows = [self._components_to_row(e) for e in aq_payload.get("list", [])]
        aq_df = pd.DataFrame(aq_rows)
        if not aq_df.empty:
            aq_df = aq_df[aq_df["datetime"] <= cutoff]

        weather_rows = []
        for entry in weather_payload.get("list", []):
            row = self._weather_entry_to_row(entry)
            row["datetime"] = pd.to_datetime(entry["dt"], unit="s", utc=True).tz_localize(None)
            weather_rows.append(row)
        weather_df = pd.DataFrame(weather_rows)
        if not weather_df.empty:
            weather_df = weather_df[weather_df["datetime"] <= cutoff]

        merged = pd.merge(weather_df, aq_df, on="datetime", how="outer").sort_values("datetime")
        return _attach_city_columns(merged, city)

    def fetch_historical(
        self, city: config.City, start_date: date, end_date: date
    ) -> pd.DataFrame:
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        payload = _get_json(
            config.OPENWEATHER_AIR_POLLUTION_HISTORY_URL,
            {
                "lat": city.latitude,
                "lon": city.longitude,
                "start": int(start_dt.timestamp()),
                "end": int(end_dt.timestamp()),
                "appid": self.api_key,
            },
        )
        rows = [self._components_to_row(e) for e in payload.get("list", [])]
        aq_df = pd.DataFrame(rows)

        logger.warning(
            "OpenWeather's free tier has no historical weather endpoint; "
            "weather columns for %s in [%s, %s] will be NaN. Use "
            "AQI_DATA_PROVIDER=open_meteo for full historical backfill.",
            city.name,
            start_date,
            end_date,
        )
        for var in config.WEATHER_HOURLY_VARS:
            if var not in aq_df.columns:
                aq_df[var] = None

        return _attach_city_columns(aq_df, city)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_client(provider: str | None = None) -> BaseAQIClient:
    """Return the configured `BaseAQIClient` (`AQI_DATA_PROVIDER` env var by default)."""
    provider = (provider or config.AQI_DATA_PROVIDER).strip().lower()
    if provider == "open_meteo":
        return OpenMeteoClient()
    if provider == "openweather":
        return OpenWeatherClient()
    raise ValueError(
        f"Unknown AQI_DATA_PROVIDER={provider!r}. Expected 'open_meteo' or 'openweather'."
    )
