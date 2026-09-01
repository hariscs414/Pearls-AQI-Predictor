"""
Unit tests for aqi_predictor.data.api_client, using `responses` to mock HTTP
calls with payloads shaped exactly like the real Open-Meteo / OpenWeather
API documentation examples (see module docstrings in api_client.py for the
source URLs), so these tests catch schema-parsing regressions without
hitting the network.
"""

from __future__ import annotations

import pytest
import responses

from aqi_predictor import config
from aqi_predictor.data.api_client import (
    DataFetchError,
    OpenMeteoClient,
    OpenWeatherClient,
    get_client,
)

CITY = config.City("testcity", "Test City", "Testland", 52.52, 13.41)


def _weather_hourly_payload(times: list[str]) -> dict:
    return {
        "latitude": 52.52,
        "longitude": 13.419,
        "timezone": "GMT",
        "hourly": {
            "time": times,
            "temperature_2m": [20.0] * len(times),
            "relative_humidity_2m": [55.0] * len(times),
            "dew_point_2m": [12.0] * len(times),
            "apparent_temperature": [19.5] * len(times),
            "surface_pressure": [1012.0] * len(times),
            "precipitation": [0.0] * len(times),
            "cloud_cover": [30.0] * len(times),
            "wind_speed_10m": [10.0] * len(times),
            "wind_direction_10m": [180.0] * len(times),
            "wind_gusts_10m": [15.0] * len(times),
        },
    }


def _air_quality_hourly_payload(times: list[str]) -> dict:
    return {
        "latitude": 52.52,
        "longitude": 13.419,
        "timezone": "GMT",
        "hourly": {
            "time": times,
            "us_aqi": [42.0] * len(times),
            "pm10": [10.0] * len(times),
            "pm2_5": [6.0] * len(times),
            "carbon_monoxide": [220.0] * len(times),
            "nitrogen_dioxide": [15.0] * len(times),
            "sulphur_dioxide": [4.0] * len(times),
            "ozone": [45.0] * len(times),
            "dust": [1.0] * len(times),
        },
    }


class TestOpenMeteoClient:
    @responses.activate
    def test_fetch_forecast_merges_weather_and_air_quality(self):
        times = ["2026-01-01T00:00", "2026-01-01T01:00"]
        responses.add(
            responses.GET, config.OPEN_METEO_FORECAST_URL,
            json=_weather_hourly_payload(times), status=200,
        )
        responses.add(
            responses.GET, config.OPEN_METEO_AIR_QUALITY_URL,
            json=_air_quality_hourly_payload(times), status=200,
        )

        client = OpenMeteoClient()
        df = client.fetch_forecast(CITY, forecast_days=1)

        assert len(df) == 2
        assert df["city_key"].iloc[0] == "testcity"
        assert (df["us_aqi"] == 42.0).all()
        assert (df["temperature_2m"] == 20.0).all()

    @responses.activate
    def test_fetch_historical_uses_archive_and_air_quality_endpoints(self):
        times = ["2025-06-01T00:00", "2025-06-01T01:00", "2025-06-01T02:00"]
        responses.add(
            responses.GET, config.OPEN_METEO_ARCHIVE_URL,
            json=_weather_hourly_payload(times), status=200,
        )
        responses.add(
            responses.GET, config.OPEN_METEO_AIR_QUALITY_URL,
            json=_air_quality_hourly_payload(times), status=200,
        )

        client = OpenMeteoClient()
        df = client.fetch_historical(CITY, __import__("datetime").date(2025, 6, 1),
                                      __import__("datetime").date(2025, 6, 1))
        assert len(df) == 3
        assert (df["us_aqi"] == 42.0).all()

    @responses.activate
    def test_fetch_historical_survives_missing_air_quality_data(self):
        """Older dates outside CAMS coverage -> air-quality call errors, weather still returned."""
        times = ["2018-01-01T00:00"]
        responses.add(
            responses.GET, config.OPEN_METEO_ARCHIVE_URL,
            json=_weather_hourly_payload(times), status=200,
        )
        responses.add(
            responses.GET, config.OPEN_METEO_AIR_QUALITY_URL,
            json={"error": True, "reason": "no data before 2022-08"}, status=400,
        )

        client = OpenMeteoClient()
        df = client.fetch_historical(CITY, __import__("datetime").date(2018, 1, 1),
                                      __import__("datetime").date(2018, 1, 1))
        assert len(df) == 1
        assert df["temperature_2m"].iloc[0] == 20.0
        assert df["us_aqi"].isna().all()

    @responses.activate
    def test_fetch_current_parses_current_blocks(self):
        responses.add(
            responses.GET, config.OPEN_METEO_FORECAST_URL,
            json={"current": {"time": "2026-01-01T12:00", "temperature_2m": 21.0}},
            status=200,
        )
        responses.add(
            responses.GET, config.OPEN_METEO_AIR_QUALITY_URL,
            json={"current": {"time": "2026-01-01T12:00", "us_aqi": 55.0}},
            status=200,
        )

        client = OpenMeteoClient()
        df = client.fetch_current(CITY)
        assert len(df) == 1
        assert df["us_aqi"].iloc[0] == 55.0
        assert df["temperature_2m"].iloc[0] == 21.0

    @responses.activate
    def test_server_error_raises_data_fetch_error_after_retries(self):
        responses.add(responses.GET, config.OPEN_METEO_FORECAST_URL,
                       json={"error": True, "reason": "boom"}, status=500)
        client = OpenMeteoClient()
        with pytest.raises(DataFetchError):
            client._fetch_weather_hourly(CITY, historical=False, forecast_days=1)


class TestOpenWeatherClient:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.setattr(config, "OPENWEATHER_API_KEY", "")
        with pytest.raises(ValueError):
            OpenWeatherClient(api_key="")

    @responses.activate
    def test_fetch_current_derives_aqi_from_components(self):
        responses.add(
            responses.GET, config.OPENWEATHER_AIR_POLLUTION_URL,
            json={
                "coord": {"lon": 13.41, "lat": 52.52},
                "list": [
                    {
                        "main": {"aqi": 2},
                        "components": {
                            "co": 270.367, "no": 5.867, "no2": 43.184, "o3": 4.783,
                            "so2": 14.544, "pm2_5": 13.448, "pm10": 15.524, "nh3": 0.289,
                        },
                        "dt": 1606482000,
                    }
                ],
            },
            status=200,
        )
        responses.add(
            responses.GET, "https://api.openweathermap.org/data/2.5/weather",
            json={
                "dt": 1606482000,
                "main": {"temp": 5.2, "humidity": 80, "pressure": 1005, "feels_like": 3.1},
                "wind": {"speed": 4.1, "deg": 220},
                "clouds": {"all": 90},
            },
            status=200,
        )

        client = OpenWeatherClient(api_key="dummy")
        df = client.fetch_current(CITY)
        assert len(df) == 1
        assert df["us_aqi"].iloc[0] is not None
        assert df["us_aqi"].iloc[0] > 0
        assert df["temperature_2m"].iloc[0] == 5.2


def test_get_client_factory_selects_provider(monkeypatch):
    monkeypatch.setattr(config, "AQI_DATA_PROVIDER", "open_meteo")
    assert isinstance(get_client(), OpenMeteoClient)

    with pytest.raises(ValueError):
        get_client(provider="not_a_real_provider")
