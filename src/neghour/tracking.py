"""
Experiment tracking with MLflow.

With DAGSHUB_TOKEN set, runs go to the MLflow server of the DagsHub repo
DAGSHUB_REPO. Without it, MLflow uses MLFLOW_TRACKING_URI if set, otherwise its
local default, so a run on a laptop still keeps its history. MLflow tracks and
compares runs; it does not schedule them.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol

import pandas as pd

from neghour import config

EXPERIMENT = "neghour"


class Tracker(Protocol):
    def log(
        self,
        run_name: str,
        params: dict[str, object],
        metrics: dict[str, float],
        tables: dict[str, pd.DataFrame],
    ) -> str | None:
        """Record one run and return its id, if the backend has one."""


class NullTracker:
    """Records nothing. Used by tests and dry runs."""

    def log(
        self,
        run_name: str,
        params: dict[str, object],
        metrics: dict[str, float],
        tables: dict[str, pd.DataFrame],
    ) -> str | None:
        return None


class MlflowTracker:
    """Logs parameters, metrics and CSV tables to MLflow."""

    def __init__(self) -> None:
        import mlflow

        if config.DAGSHUB_TOKEN:
            owner = config.DAGSHUB_REPO.split("/")[0]
            os.environ.setdefault("MLFLOW_TRACKING_USERNAME", owner)
            os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", config.DAGSHUB_TOKEN)
            mlflow.set_tracking_uri(f"https://dagshub.com/{config.DAGSHUB_REPO}.mlflow")
        mlflow.set_experiment(EXPERIMENT)
        self._mlflow = mlflow

    def log(
        self,
        run_name: str,
        params: dict[str, object],
        metrics: dict[str, float],
        tables: dict[str, pd.DataFrame],
    ) -> str | None:
        mlflow = self._mlflow
        finite = {k: v for k, v in metrics.items() if pd.notna(v)}
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(finite)
            with tempfile.TemporaryDirectory() as tmp:
                for name, table in tables.items():
                    path = Path(tmp) / f"{name}.csv"
                    table.to_csv(path, index=False)
                    mlflow.log_artifact(str(path))
            return run.info.run_id
