"""
LightGBM model for negative-price binary classification.

Target: P(DE-LU day-ahead price < 0) for each delivery hour.

Handling the rare positive class (~3–6% of hours):
  - scale_pos_weight = n_negative / n_positive balances the loss
  - Decision threshold is chosen on the validation set to maximise F1
    (default 0.5 is suboptimal for rare-event classification)

Serialisation: joblib, registered in Hopsworks Model Registry.
Feature importance logged to MLflow as a bar-chart artifact.

Hyperparameter tuning: Optuna (optional, post-MVP).
"""

from __future__ import annotations

import pandas as pd

# TODO: implement class NegHourLGBM:
#     fit(X_train, y_train, X_val, y_val) -> self
#       sets scale_pos_weight automatically from y_train
#     predict_proba(X) -> np.ndarray  (probabilities for positive class)
#     predict(X, threshold: float | None = None) -> np.ndarray
#       uses self.threshold_ if threshold is None
#     threshold_from_validation(X_val, y_val) -> float
#       grid-searches threshold in [0.1, 0.9] to maximise F1
#     feature_importance() -> pd.DataFrame
#       columns: feature, importance_gain, importance_split

# TODO: implement save(path: str) -> None  (joblib.dump)
# TODO: implement load(path: str) -> NegHourLGBM  (joblib.load, classmethod)
