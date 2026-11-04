"""
Hopsworks backend.

Features and labels live in separate feature groups, because labels refresh on
every run while features stay as they were at the cut-off. The primary key is
`delivery_epoch`: the UTC delivery hour as integer seconds. `delivery_utc` is
the event time.
"""

from __future__ import annotations

from typing import Any

import hopsworks
import pandas as pd

from neghour import config

KEY_COLUMN = "delivery_epoch"
EPOCH = pd.Timestamp(0, tz="UTC")


def _with_key(rows: pd.DataFrame) -> pd.DataFrame:
    keyed = rows.copy()
    keyed[KEY_COLUMN] = (keyed["delivery_utc"] - EPOCH) // pd.Timedelta(seconds=1)
    return keyed


class HopsworksStore:
    """Stores features and labels in the configured Hopsworks project."""

    def __init__(self) -> None:
        project = hopsworks.login(
            project=config.HOPSWORKS_PROJECT, api_key_value=config.HOPSWORKS_API_KEY
        )
        self._store = project.get_feature_store()

    def _group(self, name: str, version: int, description: str) -> Any:
        return self._store.get_or_create_feature_group(
            name=name,
            version=version,
            primary_key=[KEY_COLUMN],
            event_time="delivery_utc",
            online_enabled=False,
            description=description,
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
        group = self._features_group()
        group.insert(_with_key(features), write_options={"wait_for_job": True})

    def upsert_labels(self, labels: pd.DataFrame) -> None:
        group = self._labels_group()
        group.insert(_with_key(labels), write_options={"wait_for_job": True})

    def read_features(self) -> pd.DataFrame:
        return self._features_group().read().drop(columns=KEY_COLUMN)

    def read_labels(self) -> pd.DataFrame:
        return self._labels_group().read().drop(columns=KEY_COLUMN)
