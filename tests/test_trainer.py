"""Unit tests for aqi_predictor.models.trainer."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from aqi_predictor.models.trainer import (
    PersistenceBaseline,
    evaluate,
    time_based_split,
    train_and_select_best,
)


def test_evaluate_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    metrics = evaluate(y, y.copy())
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)


def test_evaluate_known_error():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    metrics = evaluate(y_true, y_pred)
    assert metrics.rmse == pytest.approx(np.sqrt((9 + 16) / 2))
    assert metrics.mae == pytest.approx(3.5)


def test_persistence_baseline_predicts_reference_column():
    baseline = PersistenceBaseline(reference_column="us_aqi_mean")
    X = pd.DataFrame({"us_aqi_mean": [10.0, 20.0, 30.0]})
    baseline.fit(X, X["us_aqi_mean"])
    preds = baseline.predict(X)
    assert list(preds) == [10.0, 20.0, 30.0]


def test_time_based_split_is_chronological_and_respects_fraction():
    dates = [date.today() - timedelta(days=d) for d in range(20, 0, -1)]
    df = pd.DataFrame({"date": dates, "value": range(20)})
    train_df, test_df = time_based_split(df, test_fraction=0.2)

    assert len(train_df) + len(test_df) == len(df)
    assert train_df["date"].max() < test_df["date"].min()
    # ~20% of 20 dates = 4
    assert len(test_df) == 4


def test_time_based_split_raises_with_too_few_dates():
    df = pd.DataFrame({"date": [date.today()], "value": [1]})
    with pytest.raises(ValueError):
        time_based_split(df)


def test_train_and_select_best_picks_a_strong_model_over_persistence():
    rng = np.random.default_rng(0)
    n = 60
    feat1 = rng.uniform(-5, 5, n)
    feat2 = rng.uniform(-5, 5, n)
    # target is a clean linear function of the features and NOT of us_aqi_mean,
    # so a fitted linear model should clearly beat the naive persistence baseline.
    target = 2 * feat1 - 3 * feat2 + rng.normal(0, 0.05, n)
    us_aqi_mean = rng.uniform(0, 500, n)  # unrelated "current AQI" reference column

    df = pd.DataFrame(
        {
            "date": [date.today() - timedelta(days=n - i) for i in range(n)],
            "feat1": feat1,
            "feat2": feat2,
            "us_aqi_mean": us_aqi_mean,
            "target_1d": target,
        }
    )
    train_df, test_df = time_based_split(df, test_fraction=0.3)

    result = train_and_select_best(
        train_df, test_df, feature_columns=["feat1", "feat2", "us_aqi_mean"],
        target_column="target_1d", horizon=1,
    )

    assert result.best.name != "persistence"
    assert result.best.metrics.rmse < 1.0
    # leaderboard is sorted ascending by RMSE
    rmses = [c.metrics.rmse for c in result.leaderboard]
    assert rmses == sorted(rmses)
    # persistence is present for reference but was never selectable as "best"
    persistence_entries = [c for c in result.leaderboard if c.name == "persistence"]
    assert len(persistence_entries) == 1
    assert persistence_entries[0].selectable is False
