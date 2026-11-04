"""Storage interface. A backend change only touches neghour.storage.get_store."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class FeatureStore(Protocol):
    """Keyed storage for features and labels. Writes are idempotent upserts."""

    def upsert_features(self, features: pd.DataFrame) -> None:
        """Insert or replace feature rows by delivery_utc."""

    def upsert_labels(self, labels: pd.DataFrame) -> None:
        """Insert or replace label rows by delivery_utc."""

    def read_features(self) -> pd.DataFrame:
        """Return all feature rows."""

    def read_labels(self) -> pd.DataFrame:
        """Return all label rows."""
