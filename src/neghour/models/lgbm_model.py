"""
LightGBM model for negative-price binary classification.

Target: P(DE-LU day-ahead price < 0) for each delivery hour.

Inputs are MODEL_FEATURES: the stored feature values only. Keys, cut-off and
`available_at*` columns are bookkeeping. The `missing_*` flags are left out as
well: LightGBM handles NaN itself, and a flag would let the model learn source
outages as a signal.

Handling the rare positive class (about 6 % of hours):
  - scale_pos_weight = n_negative / n_positive balances the loss.
  - The decision threshold is chosen on the validation set to maximise F1,
    because 0.5 suits balanced classes, not rare events.

With a validation set, training stops early on validation average precision
(PR-AUC). Serialisation: joblib.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score

MODEL_FEATURES: list[str] = [
    # price
    "price_h_d1",
    "price_h_d7",
    "neg_hours_last_7d",
    # weather
    "irradiance",
    "cloud_cover",
    "wind_100m",
    "temperature",
    # actuals
    "load_d2",
    "solar_d2",
    "wind_d2",
    "load_h_d7",
    "residual_load_proxy",
    # calendar
    "hour_of_day",
    "weekday",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
    "holiday_share",
    "is_holiday",
]

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 50
THRESHOLD_GRID = np.round(np.arange(0.10, 0.901, 0.01), 2)
DEFAULT_THRESHOLD = 0.5


def positive_weight(y: pd.Series | np.ndarray) -> float:
    """Return n_negative / n_positive for a 0/1 label."""
    labels = np.asarray(y)
    positives = int((labels == 1).sum())
    if positives == 0:
        raise ValueError("The training labels contain no negative-price hour")
    return float((labels == 0).sum() / positives)


class NegHourLGBM:
    """LightGBM classifier with class weighting and an F1-tuned threshold."""

    def __init__(
        self,
        features: Sequence[str] = MODEL_FEATURES,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.features = list(features)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.threshold_ = DEFAULT_THRESHOLD
        self.scale_pos_weight_: float | None = None
        self.model_: LGBMClassifier | None = None

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.features].astype("float64")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> NegHourLGBM:
        """Fit; with validation data, also stop early and tune the threshold."""
        self.scale_pos_weight_ = positive_weight(y_train)
        self.model_ = LGBMClassifier(
            **self.params, scale_pos_weight=self.scale_pos_weight_
        )
        options: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            options = {
                "eval_X": (self._matrix(X_val),),
                "eval_y": (np.asarray(y_val),),
                "eval_metric": "average_precision",
                "callbacks": [early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            }
        self.model_.fit(self._matrix(X_train), np.asarray(y_train), **options)
        if X_val is not None and y_val is not None:
            self.threshold_ = self.threshold_from_validation(X_val, y_val)
        return self

    def _fitted(self) -> LGBMClassifier:
        if self.model_ is None:
            raise RuntimeError("The model is not fitted")
        return self.model_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(price < 0) per row."""
        return self._fitted().predict_proba(self._matrix(X))[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Return 0 or 1 per row, using the tuned threshold unless one is given."""
        cut = self.threshold_ if threshold is None else threshold
        return (self.predict_proba(X) >= cut).astype("int64")

    def threshold_from_validation(
        self, X_val: pd.DataFrame, y_val: pd.Series
    ) -> float:
        """Return the threshold in [0.10, 0.90] with the best validation F1.

        Ties go to the lowest threshold, which favours recall.
        """
        scores = self.predict_proba(X_val)
        labels = np.asarray(y_val)
        f1 = [
            f1_score(labels, (scores >= cut).astype("int64"), zero_division=0)
            for cut in THRESHOLD_GRID
        ]
        return float(THRESHOLD_GRID[int(np.argmax(f1))])

    def feature_importance(self) -> pd.DataFrame:
        """Return gain and split importance per feature, highest gain first."""
        booster = self._fitted().booster_
        return (
            pd.DataFrame(
                {
                    "feature": self.features,
                    "importance_gain": booster.feature_importance("gain"),
                    "importance_split": booster.feature_importance("split"),
                }
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> NegHourLGBM:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not hold a {cls.__name__}")
        return model
