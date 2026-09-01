"""Shared logging configuration.

Every entry-point script calls `get_logger(__name__)` to get a module-scoped
logger that writes structured, timestamped lines to stdout (picked up by
GitHub Actions / any process supervisor) at the level configured via the
`LOG_LEVEL` environment variable.
"""

from __future__ import annotations

import logging
import sys

from aqi_predictor.config import LOG_LEVEL

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for `name` (typically `__name__`)."""
    _configure_root()
    return logging.getLogger(name)
