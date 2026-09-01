"""
Feature store abstraction.

Two backends implement the same `FeatureStore` interface:

- `LocalFeatureStore` (default): a single Parquet file on disk under
  `data/local_store/feature_store/`. Zero setup, works offline, and is
  exactly what makes this project runnable the moment you `pip install`.
  Writes are idempotent upserts keyed on `(city_key, date)`, so re-running
  the hourly pipeline or an overlapping backfill never creates duplicates.

- `HopsworksFeatureStore` (optional): the managed, genuinely-serverless
  feature store suggested by the project brief. Activated automatically
  when `FEATURE_STORE_BACKEND=hopsworks` and `HOPSWORKS_API_KEY` are set.

`get_feature_store()` is the single entry point the rest of the codebase
uses; it never needs to know which backend is active.
"""

from __future__ import annotations

import abc

import pandas as pd

from aqi_predictor import config
from aqi_predictor.utils.logging_config import get_logger

logger = get_logger(__name__)

_PRIMARY_KEY = ["city_key", "date"]


class FeatureStore(abc.ABC):
    """Common interface for reading/writing the daily AQI feature table."""

    @abc.abstractmethod
    def write_features(self, df: pd.DataFrame) -> int:
        """Upsert `df` (keyed on city_key + date). Returns the number of rows stored."""

    @abc.abstractmethod
    def read_features(
        self,
        city_keys: list[str] | None = None,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Read features, optionally filtered by city and/or date range."""

    def read_latest(self, city_keys: list[str] | None = None) -> pd.DataFrame:
        """Convenience: the most recent row per city."""
        df = self.read_features(city_keys=city_keys)
        if df.empty:
            return df
        idx = df.groupby("city_key")["date"].idxmax()
        return df.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Local (default) backend
# ---------------------------------------------------------------------------
class LocalFeatureStore(FeatureStore):
    def __init__(self, path=None):
        self.path = path or (config.FEATURE_STORE_DIR / "aqi_features.parquet")

    def _read_raw(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self.path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def write_features(self, df: pd.DataFrame) -> int:
        if df.empty:
            logger.info("write_features called with an empty DataFrame; nothing to store.")
            return 0

        incoming = df.copy()
        incoming["date"] = pd.to_datetime(incoming["date"]).dt.date

        existing = self._read_raw()
        if existing.empty:
            combined = incoming
        else:
            combined = pd.concat([existing, incoming], ignore_index=True)

        combined = combined.drop_duplicates(subset=_PRIMARY_KEY, keep="last")
        combined = combined.sort_values(["city_key", "date"]).reset_index(drop=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.path, index=False)
        logger.info(
            "Wrote %d new/updated row(s) to local feature store (%d rows total) at %s",
            len(incoming),
            len(combined),
            self.path,
        )
        return len(incoming)

    def read_features(
        self,
        city_keys: list[str] | None = None,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        df = self._read_raw()
        if df.empty:
            return df
        if city_keys:
            df = df[df["city_key"].isin(city_keys)]
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        return df.sort_values(["city_key", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hopsworks (optional, managed) backend
# ---------------------------------------------------------------------------
class HopsworksFeatureStore(FeatureStore):
    """
    Thin wrapper around a Hopsworks Feature Group.

    Requires the optional `hopsworks` package (`pip install hopsworks`) and
    `HOPSWORKS_API_KEY` / `HOPSWORKS_PROJECT_NAME` to be set. See
    `.env.example` for how to get a free-tier account and API key.
    """

    def __init__(self):
        try:
            import hopsworks  # noqa: F401  (imported for its side effect / early error)
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
        self._fs = self._project.get_feature_store()
        self._fg = self._fs.get_or_create_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
            description="Daily AQI + weather features per city",
            primary_key=_PRIMARY_KEY,
            event_time="date",
            online_enabled=True,
        )

    def write_features(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        incoming = df.copy()
        incoming["date"] = pd.to_datetime(incoming["date"])
        self._fg.insert(incoming, write_options={"wait_for_job": True})
        logger.info("Upserted %d row(s) into Hopsworks feature group '%s'", len(incoming),
                    config.FEATURE_GROUP_NAME)
        return len(incoming)

    def read_features(
        self,
        city_keys: list[str] | None = None,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        df = self._fg.read()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if city_keys:
            df = df[df["city_key"].isin(city_keys)]
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        return df.sort_values(["city_key", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_feature_store() -> FeatureStore:
    """Return the configured `FeatureStore` (`FEATURE_STORE_BACKEND` env var)."""
    backend = config.FEATURE_STORE_BACKEND

    if backend == "hopsworks":
        if not config.HOPSWORKS_API_KEY:
            logger.warning(
                "FEATURE_STORE_BACKEND=hopsworks but HOPSWORKS_API_KEY is not set; "
                "falling back to the local Parquet feature store."
            )
            return LocalFeatureStore()
        try:
            return HopsworksFeatureStore()
        except Exception:
            logger.exception(
                "Failed to connect to Hopsworks; falling back to the local Parquet "
                "feature store so the pipeline can still complete."
            )
            return LocalFeatureStore()

    return LocalFeatureStore()
