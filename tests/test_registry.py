"""Unit tests for aqi_predictor.models.registry.LocalModelRegistry."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from aqi_predictor.models.registry import LocalModelRegistry


@pytest.fixture
def fitted_ridge():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 1.0]])
    y = np.array([5.0, 4.0, 9.0, 6.0])
    model = Ridge(alpha=1.0).fit(X, y)
    return model, X


def test_has_model_false_before_saving(tmp_path):
    registry = LocalModelRegistry(directory=tmp_path)
    assert registry.has_model(1) is False


def test_save_and_load_roundtrip_preserves_predictions(tmp_path, fitted_ridge):
    model, X = fitted_ridge
    registry = LocalModelRegistry(directory=tmp_path)

    registry.save_model(
        model=model, horizon=1, model_type="ridge",
        metrics={"rmse": 1.23, "mae": 0.9, "r2": 0.95},
        feature_columns=["a", "b"],
    )
    assert registry.has_model(1) is True

    loaded_model, metadata = registry.load_model(1)
    np.testing.assert_allclose(loaded_model.predict(X), model.predict(X))
    assert metadata["model_type"] == "ridge"
    assert metadata["feature_columns"] == ["a", "b"]
    assert metadata["metrics"]["rmse"] == pytest.approx(1.23)


def test_load_model_raises_clear_error_when_missing(tmp_path):
    registry = LocalModelRegistry(directory=tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.load_model(3)


def test_different_horizons_are_independent(tmp_path, fitted_ridge):
    model, _ = fitted_ridge
    registry = LocalModelRegistry(directory=tmp_path)
    registry.save_model(model, 1, "ridge", {"rmse": 1.0, "mae": 1.0, "r2": 1.0}, ["a", "b"])
    assert registry.has_model(1) is True
    assert registry.has_model(2) is False


def test_training_history_is_appended(tmp_path, fitted_ridge):
    model, _ = fitted_ridge
    registry = LocalModelRegistry(directory=tmp_path)
    registry.save_model(model, 1, "ridge", {"rmse": 1.0, "mae": 1.0, "r2": 1.0}, ["a"])
    registry.save_model(model, 1, "ridge", {"rmse": 0.5, "mae": 0.5, "r2": 0.99}, ["a"])

    history_path = tmp_path / "training_history.jsonl"
    lines = history_path.read_text().strip().split("\n")
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["metrics"]["rmse"] == 1.0
    assert second["metrics"]["rmse"] == 0.5
    # the "active" model for horizon=1 is the most recently saved one
    _, latest_meta = registry.load_model(1)
    assert latest_meta["metrics"]["rmse"] == 0.5
