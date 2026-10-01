"""
Naive baseline models used as lower bounds in the success criterion.

NaiveYesterday:
  For each delivery hour h on D, predicts y_neg = label(h, D-1).
  Rationale: if yesterday's hour was negative, tomorrow's may be too.

NaiveLastWeek:
  For each delivery hour h on D, predicts y_neg = label(h, D-7).
  Rationale: weather and demand patterns repeat weekly.

Success criterion (proposal §1):
  F1(LightGBM) >= max(F1(NaiveYesterday), F1(NaiveLastWeek)) + 0.10
"""

from __future__ import annotations

import pandas as pd

# TODO: define BaselineModel protocol with fit(df: pd.DataFrame) and
#       predict(df: pd.DataFrame) -> pd.Series methods

# TODO: implement class NaiveYesterday(BaselineModel)
#       stores {(delivery_hour, D-1): y_neg} during fit
#       looks up same hour 1 day before delivery_date during predict

# TODO: implement class NaiveLastWeek(BaselineModel)
#       stores {(delivery_hour, D-7): y_neg} during fit
#       looks up same hour 7 days before delivery_date during predict

# TODO: both classes must handle missing look-back dates gracefully
#       (default prediction = 0 / majority class)
