"""
Model registry abstraction.

Mirrors `aqi_predictor.features.feature_store`: a zero-setup local backend
(joblib files + JSON metadata under `data/local_store/model_registry/`) by
default, and an optional Hopsworks Model Registry backend for the fully
managed / serverless deployment described in the project brief.

One model is trained and registered per forecast horizon (day+1, day+2,
day+3 -- see `config.FORECAST_HORIZON_DAYS`), since a 3-step-ahead forecast
is most accurately produced as three specialised regressors rather than one
model asked to predict three different things at once (see REPORT.md for
the reasoning and the alternative considered).
"""

from __future__ import annotations

import abc
import json
from datetime import datetime, timezone
from typing import Any

import joblib

from aqi_predictor import config
from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelRegistry(abc.ABC):
    @abc.abstractmethod
    def save_model(
        self,
        model: Any,
        horizon: int,
        model_type: str,
        metrics: dict,
        feature_columns: list[str],
    ) -> str:
        """Persist `model` as the active model for `horizon`. Returns a version id."""

    @abc.abstractmethod
    def load_model(self, horizon: int) -> tuple[Any, dict]:
        """Return `(model, metadata)` for the active model at `horizon`."""

    @abc.abstractmethod
    def has_model(self, horizon: int) -> bool:
        ...


# ---------------------------------------------------------------------------
# Local (default) backend
# ---------------------------------------------------------------------------
class LocalModelRegistry(ModelRegistry):
    def __init__(self, directory=None):
        self.directory = directory or config.MODEL_REGISTRY_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def _model_path(self, horizon: int):
        return self.directory / f"{config.MODEL_NAME}_h{horizon}d.joblib"

    def _meta_path(self, horizon: int):
        return self.directory / f"{config.MODEL_NAME}_h{horizon}d_meta.json"

    def _history_path(self):
        return self.directory / "training_history.jsonl"

    def save_model(
        self,
        model: Any,
        horizon: int,
        model_type: str,
        metrics: dict,
        feature_columns: list[str],
    ) -> str:
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        metadata = {
            "version": version,
            "horizon_days": horizon,
            "model_type": model_type,
            "metrics": metrics,
            "feature_columns": feature_columns,
            "trained_at": version,
        }

        joblib.dump(model, self._model_path(horizon))
        self._meta_path(horizon).write_text(json.dumps(metadata, indent=2))

        with open(self._history_path(), "a") as f:
            f.write(json.dumps(metadata) + "\n")

        logger.info(
            "Registered model for horizon=%dd: type=%s RMSE=%.3f MAE=%.3f R2=%.3f",
            horizon,
            model_type,
            metrics.get("rmse", float("nan")),
            metrics.get("mae", float("nan")),
            metrics.get("r2", float("nan")),
        )
        return version

    def load_model(self, horizon: int) -> tuple[Any, dict]:
        if not self.has_model(horizon):
            raise FileNotFoundError(
                f"No registered model for horizon={horizon}d. Run the training "
                f"pipeline first: python scripts/run_training_pipeline.py"
            )
        model = joblib.load(self._model_path(horizon))
        metadata = json.loads(self._meta_path(horizon).read_text())
        return model, metadata

    def has_model(self, horizon: int) -> bool:
        return self._model_path(horizon).exists() and self._meta_path(horizon).exists()


# ---------------------------------------------------------------------------
# Hopsworks (optional, managed) backend
# ---------------------------------------------------------------------------
class HopsworksModelRegistry(ModelRegistry):
    """Thin wrapper around a Hopsworks Model Registry entry per horizon."""

    def __init__(self):
        try:
            import hopsworks  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FEATURE_STORE_BACKEND=hopsworks requires the optional 'hopsworks' "
                "package: pip install hopsworks"
            ) from exc

        import hopsworks

        self._project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT_NAME or None,
        )
        self._mr = self._project.get_model_registry()
        # A local scratch directory is still needed: Hopsworks models are
        # uploaded as a directory of files (joblib blob + metadata.json).
        self._scratch = config.MODEL_REGISTRY_DIR / "_hopsworks_scratch"
        self._scratch.mkdir(parents=True, exist_ok=True)

    def _name(self, horizon: int) -> str:
        return f"{config.MODEL_NAME}_h{horizon}d"

    def save_model(
        self,
        model: Any,
        horizon: int,
        model_type: str,
        metrics: dict,
        feature_columns: list[str],
    ) -> str:
        model_dir = self._scratch / self._name(horizon)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        (model_dir / "metadata.json").write_text(
            json.dumps({"model_type": model_type, "feature_columns": feature_columns}, indent=2)
        )

        hw_model = self._mr.python.create_model(
            name=self._name(horizon),
            metrics={k: float(v) for k, v in metrics.items()},
            description=f"AQI forecaster, {horizon}-day-ahead ({model_type})",
        )
        hw_model.save(str(model_dir))
        logger.info("Registered model '%s' (v%s) in Hopsworks Model Registry",
                    self._name(horizon), hw_model.version)
        return str(hw_model.version)

    def load_model(self, horizon: int) -> tuple[Any, dict]:
        hw_model = self._mr.get_best_model(self._name(horizon), "rmse", "min")
        if hw_model is None:
            raise FileNotFoundError(f"No registered model '{self._name(horizon)}' in Hopsworks")
        model_dir = hw_model.download()
        model = joblib.load(f"{model_dir}/model.joblib")
        metadata = json.loads(open(f"{model_dir}/metadata.json").read())
        metadata["metrics"] = hw_model.training_metrics
        metadata["version"] = hw_model.version
        return model, metadata

    def has_model(self, horizon: int) -> bool:
        try:
            return self._mr.get_best_model(self._name(horizon), "rmse", "min") is not None
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_model_registry() -> ModelRegistry:
    """Return the configured `ModelRegistry` (mirrors `get_feature_store()`)."""
    backend = config.FEATURE_STORE_BACKEND  # one backend switch governs both stores

    if backend == "hopsworks":
        if not config.HOPSWORKS_API_KEY:
            logger.warning(
                "FEATURE_STORE_BACKEND=hopsworks but HOPSWORKS_API_KEY is not set; "
                "falling back to the local model registry."
            )
            return LocalModelRegistry()
        try:
            return HopsworksModelRegistry()
        except Exception:
            logger.exception(
                "Failed to connect to Hopsworks Model Registry; falling back to the "
                "local model registry so the pipeline can still complete."
            )
            return LocalModelRegistry()

    return LocalModelRegistry()
