"""Unit tests for aqi_predictor.models.forecaster, using minimal in-memory
fakes for FeatureStore / ModelRegistry so no disk or network I/O is needed.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from aqi_predictor.features.feature_store import FeatureStore
from aqi_predictor.models.forecaster import ForecastUnavailableError, forecast_city, forecast_many
from aqi_predictor.models.registry import ModelRegistry


class FakeFeatureStore(FeatureStore):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def write_features(self, df):
        raise NotImplementedError

    def read_features(self, city_keys=None, start_date=None, end_date=None):
        df = self._df
        if city_keys:
            df = df[df["city_key"].isin(city_keys)]
        return df.reset_index(drop=True)


class ConstantModel:
    """Predicts a fixed value regardless of input, for deterministic tests."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


class FakeModelRegistry(ModelRegistry):
    def __init__(self, models: dict[int, tuple]):
        self._models = models

    def save_model(self, *a, **kw):
        raise NotImplementedError

    def load_model(self, horizon):
        return self._models[horizon]

    def has_model(self, horizon):
        return horizon in self._models


FEATURE_COLUMNS = ["us_aqi_mean", "temperature_2m_mean"]


def _feature_row(as_of: date, aqi=80.0):
    return pd.DataFrame(
        [{
            "city_key": "testcity", "city_name": "Test City", "date": as_of,
            "us_aqi_mean": aqi, "temperature_2m_mean": 22.0,
        }]
    )


def test_forecast_city_produces_correct_target_dates_and_categories():
    as_of = date.today()
    store = FakeFeatureStore(_feature_row(as_of))
    registry = FakeModelRegistry(
        {
            1: (ConstantModel(70), {"feature_columns": FEATURE_COLUMNS}),
            2: (ConstantModel(120), {"feature_columns": FEATURE_COLUMNS}),
            3: (ConstantModel(210), {"feature_columns": FEATURE_COLUMNS}),
        }
    )

    result = forecast_city("testcity", store, registry, horizon_days=3)

    assert result.city_key == "testcity"
    assert result.as_of_date == as_of
    assert [p.horizon_days for p in result.points] == [1, 2, 3]
    assert [p.target_date for p in result.points] == [
        as_of + timedelta(days=1), as_of + timedelta(days=2), as_of + timedelta(days=3)
    ]
    assert result.points[0].category.name == "Moderate"
    assert result.points[1].category.name == "Unhealthy for Sensitive Groups"
    assert result.points[2].category.name == "Very Unhealthy"


def test_forecast_city_clamps_predictions_to_valid_range():
    as_of = date.today()
    store = FakeFeatureStore(_feature_row(as_of))
    registry = FakeModelRegistry(
        {1: (ConstantModel(-50), {"feature_columns": FEATURE_COLUMNS})}
    )
    result = forecast_city("testcity", store, registry, horizon_days=1)
    assert result.points[0].predicted_aqi == 0.0

    registry = FakeModelRegistry(
        {1: (ConstantModel(999), {"feature_columns": FEATURE_COLUMNS})}
    )
    result = forecast_city("testcity", store, registry, horizon_days=1)
    assert result.points[0].predicted_aqi == 500.0


def test_forecast_city_raises_when_no_features():
    store = FakeFeatureStore(pd.DataFrame(columns=["city_key", "date"]))
    registry = FakeModelRegistry({})
    with pytest.raises(ForecastUnavailableError):
        forecast_city("missing_city", store, registry)


def test_forecast_city_raises_when_model_missing():
    store = FakeFeatureStore(_feature_row(date.today()))
    registry = FakeModelRegistry({})  # no horizon=1 model registered
    with pytest.raises(ForecastUnavailableError):
        forecast_city("testcity", store, registry, horizon_days=1)


def test_forecast_many_isolates_per_city_failures():
    store = FakeFeatureStore(_feature_row(date.today()))
    registry = FakeModelRegistry(
        {1: (ConstantModel(60), {"feature_columns": FEATURE_COLUMNS})}
    )
    results = forecast_many(["testcity", "unknown_city"], store, registry, horizon_days=1)

    assert results["testcity"].points[0].predicted_aqi == 60.0
    assert isinstance(results["unknown_city"], Exception)
