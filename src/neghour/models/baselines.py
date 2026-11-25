"""
Naive baseline models used as lower bounds in the success criterion.

NaiveYesterday:
  Predicts y_neg = 1 if price_h_d1 < 0, the same hour on D-1.
  Rationale: if yesterday's hour was negative, tomorrow's may be too.

NaiveLastWeek:
  Predicts y_neg = 1 if price_h_d7 < 0, the same hour on D-7.
  Rationale: weather and demand patterns repeat weekly.

Both read the stored price features, which already follow the Berlin wall clock
across clock changes and respect the cut-off. A missing lag predicts 0, the
majority class. The label uses the same rule, price < 0, so a stored 0.00 is
not negative in either.

Success criterion (proposal section 1):
  F1(LightGBM) >= max(F1(NaiveYesterday), F1(NaiveLastWeek)) + 0.10
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd


class NaiveBaseline:
    """Predicts a negative hour when the lagged price of that hour was negative."""

    name: ClassVar[str]
    lag_column: ClassVar[str]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return 0 or 1 per row; a missing lag gives 0."""
        lag = features[self.lag_column].astype("float64")
        return (lag < 0).to_numpy(dtype="int64")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return the 0 or 1 prediction as a score, so PR-AUC can be computed."""
        return self.predict(features).astype("float64")


class NaiveYesterday(NaiveBaseline):
    name = "naive_yesterday"
    lag_column = "price_h_d1"


class NaiveLastWeek(NaiveBaseline):
    name = "naive_last_week"
    lag_column = "price_h_d7"


BASELINES: tuple[NaiveBaseline, ...] = (NaiveYesterday(), NaiveLastWeek())
