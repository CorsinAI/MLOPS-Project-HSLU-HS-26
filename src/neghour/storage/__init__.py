"""Storage backends. Pick one with the STORAGE_BACKEND setting."""

from __future__ import annotations

from neghour import config
from neghour.storage.base import FeatureStore
from neghour.storage.memory import MemoryStore


def get_store() -> FeatureStore:
    """Return the configured storage backend."""
    if config.STORAGE_BACKEND == "memory":
        return MemoryStore()
    if config.STORAGE_BACKEND == "hopsworks":
        from neghour.storage.hopsworks_store import HopsworksStore

        return HopsworksStore()
    raise ValueError(f"Unknown STORAGE_BACKEND: {config.STORAGE_BACKEND}")
