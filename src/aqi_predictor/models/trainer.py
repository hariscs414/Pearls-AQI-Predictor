"""
Model training and evaluation.

Trains several candidate regressors per forecast horizon (statistical to
deep learning, per the project brief) on a chronological train/test split,
evaluates each with RMSE / MAE / R^2, and returns the best candidate plus a
full leaderboard for the report / dashboard.

Candidates (a candidate is skipped, not an error, if its optional dependency
is unavailable):
  - `persistence`      : naive "tomorrow = today" baseline (reference only,
                          never selected as the winner -- see module note)
  - `ridge`             : linear model, statistical baseline
  - `random_forest`     : classic ensemble ML
  - `xgboost`           : gradient-boosted trees
  - `dense_nn`          : small Keras feed-forward network (only if
                          `tensorflow` is installed; skipped otherwise)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EvalMetrics:
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2}


@dataclass
class TrainedCandidate:
    name: str
    model: Any
    metrics: EvalMetrics
    selectable: bool = True  # False for reference baselines like `persistence`


@dataclass
class TrainingResult:
    horizon: int
    best: TrainedCandidate
    leaderboard: list[TrainedCandidate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reference baseline: naive persistence forecast
# ---------------------------------------------------------------------------
class PersistenceBaseline:
    """Predicts that AQI stays the same as the most recent known value.

    A legitimate, picklable estimator (not just a metric) so it can sit in
    the same leaderboard as the real models: if a trained model can't beat
    this, that is itself an important, reportable finding.
    """

    def __init__(self, reference_column: str = "us_aqi_mean"):
        self.reference_column = reference_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.reference_column].to_numpy()


# ---------------------------------------------------------------------------
# Optional deep-learning candidate
# ---------------------------------------------------------------------------
class KerasDenseRegressor:
    """Small feed-forward network with internal feature scaling.

    Only instantiated if `tensorflow` is importable; see `get_candidates()`.
    Implements a scikit-learn-compatible `fit`/`predict` so it can be used
    interchangeably with the other candidates (and pickled via joblib for
    the model registry, as long as TensorFlow is installed wherever the
    model is later loaded).
    """

    def __init__(self, epochs: int = 100, batch_size: int = 16, patience: int = 10):
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self._scaler = StandardScaler()
        self._model = None

    def _build(self, n_features: int):
        from tensorflow import keras

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(n_features,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.15),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KerasDenseRegressor":
        from tensorflow import keras

        X_scaled = self._scaler.fit_transform(X.to_numpy())
        self._model = self._build(X_scaled.shape[1])
        early_stop = keras.callbacks.EarlyStopping(
            monitor="loss", patience=self.patience, restore_best_weights=True
        )
        self._model.fit(
            X_scaled,
            y.to_numpy(),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[early_stop],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self._scaler.transform(X.to_numpy())
        return self._model.predict(X_scaled, verbose=0).ravel()

    def __getstate__(self):
        # Keras models aren't picklable via plain joblib out of the box;
        # serialise to in-memory HDF5 bytes instead.
        import io

        state = self.__dict__.copy()
        if self._model is not None:
            buf = io.BytesIO()
            self._model.save(buf, save_format="h5")
            state["_model_bytes"] = buf.getvalue()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        import io

        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        if model_bytes is not None:
            from tensorflow import keras

            self._model = keras.models.load_model(io.BytesIO(model_bytes))
        else:
            self._model = None


def is_tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------
def get_candidates() -> dict[str, tuple[Any, bool]]:
    """Return {name: (unfitted estimator, selectable)} for every available candidate."""
    candidates: dict[str, tuple[Any, bool]] = {
        "persistence": (PersistenceBaseline(), False),
        "ridge": (
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
            True,
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            True,
        ),
    }

    try:
        from xgboost import XGBRegressor

        candidates["xgboost"] = (
            XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            True,
        )
    except ImportError:
        logger.info("xgboost not installed; skipping the 'xgboost' candidate.")

    if is_tensorflow_available():
        candidates["dense_nn"] = (KerasDenseRegressor(), True)
    else:
        logger.info(
            "tensorflow not installed; skipping the optional 'dense_nn' candidate. "
            "Install it with `pip install tensorflow` to include a deep-learning model."
        )

    return candidates


# ---------------------------------------------------------------------------
# Splitting & evaluation
# ---------------------------------------------------------------------------
def time_based_split(
    df: pd.DataFrame, date_col: str = "date", test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: the most recent `test_fraction` of *dates* (not
    rows) become the test set. Using dates rather than rows keeps all
    cities' same-day data on the same side of the split.
    """
    dates = np.sort(df[date_col].unique())
    if len(dates) < 5:
        raise ValueError(
            f"Need at least 5 distinct dates to split train/test, got {len(dates)}. "
            "Run the backfill pipeline for a longer historical range first."
        )
    n_test = max(1, int(round(len(dates) * test_fraction)))
    split_date = dates[-n_test]
    train_df = df[df[date_col] < split_date].reset_index(drop=True)
    test_df = df[df[date_col] >= split_date].reset_index(drop=True)
    return train_df, test_df


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvalMetrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return EvalMetrics(rmse=rmse, mae=mae, r2=r2)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def train_and_select_best(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon: int,
) -> TrainingResult:
    """Fit every candidate, evaluate on `test_df`, and pick the best selectable one."""
    X_train, y_train = train_df[feature_columns], train_df[target_column]
    X_test, y_test = test_df[feature_columns], test_df[target_column]

    leaderboard: list[TrainedCandidate] = []
    for name, (estimator, selectable) in get_candidates().items():
        try:
            estimator.fit(X_train, y_train)
            preds = estimator.predict(X_test)
            metrics = evaluate(y_test.to_numpy(), preds)
        except Exception:
            logger.exception("Candidate '%s' failed to train/evaluate for horizon=%dd; skipping.",
                              name, horizon)
            continue
        leaderboard.append(TrainedCandidate(name, estimator, metrics, selectable))
        logger.info(
            "[h=%dd] %-14s RMSE=%.3f  MAE=%.3f  R2=%.3f",
            horizon, name, metrics.rmse, metrics.mae, metrics.r2,
        )

    selectable = [c for c in leaderboard if c.selectable]
    if not selectable:
        raise RuntimeError(f"No selectable candidate trained successfully for horizon={horizon}d")

    best = min(selectable, key=lambda c: c.metrics.rmse)
    logger.info("[h=%dd] Selected best model: %s (RMSE=%.3f)", horizon, best.name, best.metrics.rmse)

    leaderboard.sort(key=lambda c: c.metrics.rmse)
    return TrainingResult(horizon=horizon, best=best, leaderboard=leaderboard)
