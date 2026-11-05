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

Writes retry on server errors. An upsert is idempotent, so repeating one after
a failed metadata commit is safe. Client errors fail at once.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import hopsworks
import pandas as pd
from hopsworks_common.client.exceptions import RestAPIError

from neghour import config

logger = logging.getLogger(__name__)

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


def _insert(group: Any, rows: pd.DataFrame) -> None:
    """Insert rows, retrying when the server fails."""
    for attempt in range(1, WRITE_ATTEMPTS + 1):
        try:
            group.insert(rows)
            return
        except RestAPIError as error:
            status = error.response.status_code
            if status < 500 or attempt == WRITE_ATTEMPTS:
                raise
            logger.warning("Hopsworks write failed: HTTP %s, retrying", status)
            time.sleep(WRITE_RETRY_DELAY_S)


class HopsworksStore:
    """Stores features and labels in the configured Hopsworks project."""

    def __init__(self) -> None:
        self._store = login().get_feature_store()

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
