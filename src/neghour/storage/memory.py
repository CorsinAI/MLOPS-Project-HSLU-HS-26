"""In-memory store for tests and dry runs. Nothing is persisted."""

from __future__ import annotations

import pandas as pd


class MemoryStore:
    """Keeps features and labels in data frames keyed by delivery_utc."""

    def __init__(self) -> None:
        self._features = pd.DataFrame()
        self._labels = pd.DataFrame()

    @staticmethod
    def _upsert(current: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        if current.empty:
            combined = rows
        else:
            combined = pd.concat([current, rows], ignore_index=True)
        latest = combined.drop_duplicates("delivery_utc", keep="last")
        return latest.sort_values("delivery_utc").reset_index(drop=True)

    def upsert_features(self, features: pd.DataFrame) -> None:
        self._features = self._upsert(self._features, features)

    def upsert_labels(self, labels: pd.DataFrame) -> None:
        self._labels = self._upsert(self._labels, labels)

    def read_features(self) -> pd.DataFrame:
        return self._features.copy()

    def read_labels(self) -> pd.DataFrame:
        return self._labels.copy()
