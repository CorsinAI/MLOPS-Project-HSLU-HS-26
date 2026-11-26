"""
Model registry: where promoted models are kept and where inference loads them.

Only a model that met the promotion criteria is registered, so the newest
registered version is the one inference serves. Each version holds the fitted
NegHourLGBM (with its tuned threshold) as `model.joblib` and its metrics.

Backends follow STORAGE_BACKEND:
  hopsworks  the Hopsworks Model Registry of the configured project. Saving a
             model uploads files only and starts no job.
  memory     a local folder (LOCAL_MODEL_DIR), for tests and dry runs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Protocol

from neghour import config
from neghour.models.lgbm_model import NegHourLGBM

MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"


class NoModelError(LookupError):
    """Raised when inference finds no registered model."""


class ModelRegistry(Protocol):
    def register(
        self, model: NegHourLGBM, metrics: dict[str, float], description: str
    ) -> int:
        """Store a promoted model and return its version."""

    def load_latest(self) -> tuple[NegHourLGBM, int]:
        """Return the newest registered model and its version."""


def _write(folder: Path, model: NegHourLGBM, metrics: dict[str, float]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / MODEL_FILE)
    (folder / METRICS_FILE).write_text(json.dumps(metrics, indent=2), "utf-8")


class LocalRegistry:
    """Keeps versions as numbered subfolders: <root>/<name>/<version>/."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / config.MODEL_NAME

    def _versions(self) -> list[int]:
        if not self.root.exists():
            return []
        return sorted(int(p.name) for p in self.root.iterdir() if p.name.isdigit())

    def register(
        self, model: NegHourLGBM, metrics: dict[str, float], description: str
    ) -> int:
        version = (self._versions() or [0])[-1] + 1
        folder = self.root / str(version)
        _write(folder, model, metrics)
        (folder / "description.txt").write_text(description, "utf-8")
        return version

    def load_latest(self) -> tuple[NegHourLGBM, int]:
        versions = self._versions()
        if not versions:
            raise NoModelError(f"No registered model in {self.root}")
        version = versions[-1]
        return NegHourLGBM.load(self.root / str(version) / MODEL_FILE), version


class HopsworksRegistry:
    """Uses the Model Registry of the configured Hopsworks project."""

    def __init__(self) -> None:
        from neghour.storage.hopsworks_store import _with_retry, login

        self._retry = _with_retry
        self._registry: Any = _with_retry(
            lambda: login().get_model_registry(), "login"
        )

    def register(
        self, model: NegHourLGBM, metrics: dict[str, float], description: str
    ) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            _write(Path(tmp), model, metrics)
            entry = self._registry.python.create_model(
                name=config.MODEL_NAME, metrics=metrics, description=description
            )
            self._retry(lambda: entry.save(tmp), "model upload")
        return int(entry.version)

    def load_latest(self) -> tuple[NegHourLGBM, int]:
        models = self._retry(
            lambda: self._registry.get_models(config.MODEL_NAME), "read"
        )
        if not models:
            raise NoModelError(f"No registered model named {config.MODEL_NAME}")
        latest = max(models, key=lambda m: m.version)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(self._retry(lambda: latest.download(tmp), "model download"))
            return NegHourLGBM.load(folder / MODEL_FILE), int(latest.version)


def get_registry() -> ModelRegistry:
    """Return the registry that matches STORAGE_BACKEND."""
    if config.STORAGE_BACKEND == "memory":
        return LocalRegistry(config.LOCAL_MODEL_DIR)
    if config.STORAGE_BACKEND == "hopsworks":
        return HopsworksRegistry()
    raise ValueError(f"Unknown STORAGE_BACKEND: {config.STORAGE_BACKEND}")
