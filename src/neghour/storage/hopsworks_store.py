"""
Hopsworks backend.

Features and labels live in separate feature groups, because labels only exist
after the auction while features use only data available by the cut-off. Each run
rewrites its whole window of both from current source values. The primary key is
`delivery_epoch`: the UTC delivery hour as integer seconds. `delivery_utc` is
the event time.

Groups are offline only. With `stream=False` and Delta, the client writes the
table directly, so an insert starts no Spark job. Statistics are off because
they would start one after every insert. See GROUP_SETTINGS.

Login and writes retry on transient failures: HTTP 5xx, a dropped connection
or timeout, and storage I/O errors. All four failed scheduled runs before this
change were of these kinds. An upsert is idempotent and a Delta commit is
atomic, so repeating a failed write is safe. Client errors (HTTP 4xx) fail at
once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import hopsworks
import pandas as pd
from hopsworks_common.client.exceptions import RestAPIError

from neghour import config

logger = logging.getLogger(__name__)

T = TypeVar("T")

KEY_COLUMN = "delivery_epoch"
WRITE_ATTEMPTS = 3
WRITE_RETRY_DELAY_S = 30.0
EPOCH = pd.Timestamp(0, tz="UTC")
GROUP_SETTINGS: dict[str, Any] = {
    "online_enabled": False,
    "stream": False,
    "time_travel_format": "DELTA",
    "statistics_config": False,
}


def login() -> Any:
    """Log in to the configured Hopsworks instance and project."""
    host = config.HOPSWORKS_HOST.removeprefix("https://").rstrip("/")
    if not host:
        raise RuntimeError("HOPSWORKS_HOST is not set")
    return hopsworks.login(
        host=host,
        project=config.HOPSWORKS_PROJECT,
        api_key_value=config.HOPSWORKS_API_KEY,
    )


def _with_key(rows: pd.DataFrame) -> pd.DataFrame:
    keyed = rows.copy()
    keyed[KEY_COLUMN] = (keyed["delivery_utc"] - EPOCH) // pd.Timedelta(seconds=1)
    return keyed


def _transient(error: Exception) -> bool:
    """Return whether a failed Hopsworks call is worth repeating.

    requests' ConnectionError and Timeout are OSErrors, like the delta-rs
    storage error seen on 2026-11-17.
    """
    if isinstance(error, RestAPIError):
        return error.response.status_code >= 500
    return isinstance(error, OSError)


def _with_retry(call: Callable[[], T], what: str) -> T:
    """Run a Hopsworks call, repeating it on transient failures."""
    for attempt in range(1, WRITE_ATTEMPTS + 1):
        try:
            return call()
        except (RestAPIError, OSError) as error:
            if not _transient(error) or attempt == WRITE_ATTEMPTS:
                raise
            logger.warning(
                "Hopsworks %s failed (attempt %d of %d), retrying: %r",
                what, attempt, WRITE_ATTEMPTS, error,
            )
            time.sleep(WRITE_RETRY_DELAY_S)
    raise AssertionError("unreachable")


def _insert(group: Any, rows: pd.DataFrame) -> None:
    """Insert rows, retrying on transient failures."""
    _with_retry(lambda: group.insert(rows), "write")


class HopsworksStore:
    """Stores features and labels in the configured Hopsworks project."""

    def __init__(self) -> None:
        self._store = _with_retry(lambda: login().get_feature_store(), "login")

    def _group(self, name: str, version: int, description: str) -> Any:
        return self._store.get_or_create_feature_group(
            name=name,
            version=version,
            primary_key=[KEY_COLUMN],
            event_time="delivery_utc",
            description=description,
            **GROUP_SETTINGS,
        )

    def _features_group(self) -> Any:
        return self._group(
            config.FEATURE_GROUP_NAME,
            config.FEATURE_GROUP_VERSION,
            "Features per delivery hour, as known at the D-1 10:00 Berlin cut-off",
        )

    def _labels_group(self) -> Any:
        return self._group(
            config.LABEL_GROUP_NAME,
            config.LABEL_GROUP_VERSION,
            "Hourly day-ahead price and y_neg label per delivery hour",
        )

    def upsert_features(self, features: pd.DataFrame) -> None:
        _insert(self._features_group(), _with_key(features))

    def upsert_labels(self, labels: pd.DataFrame) -> None:
        _insert(self._labels_group(), _with_key(labels))

    def read_features(self) -> pd.DataFrame:
        return self._features_group().read().drop(columns=KEY_COLUMN)

    def read_labels(self) -> pd.DataFrame:
        return self._labels_group().read().drop(columns=KEY_COLUMN)
