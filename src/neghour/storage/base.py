"""Storage interface. A backend change only touches neghour.storage.get_store."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class FeatureStore(Protocol):
    """Keyed storage for features, labels and predictions.

    Writes are idempotent upserts by delivery_utc. A read of a group that was
    never written returns an empty frame.
    """

    def upsert_features(self, features: pd.DataFrame) -> None:
        """Insert or replace feature rows by delivery_utc."""

    def upsert_labels(self, labels: pd.DataFrame) -> None:
        """Insert or replace label rows by delivery_utc."""

    def read_features(self) -> pd.DataFrame:
        """Return all feature rows."""

    def read_labels(self) -> pd.DataFrame:
        """Return all label rows."""

    def upsert_predictions(self, predictions: pd.DataFrame) -> None:
        """Insert or replace prediction rows by delivery_utc."""

    def read_predictions(self) -> pd.DataFrame:
        """Return all prediction rows."""
