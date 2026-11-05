"""
Naive baseline models used as lower bounds in the success criterion.

NaiveYesterday:
  Predicts y_neg = 1 if price_h_d1 < 0, the same hour on D-1.
  Rationale: if yesterday's hour was negative, tomorrow's may be too.

NaiveLastWeek:
  Predicts y_neg = 1 if price_h_d7 < 0, the same hour on D-7.
  Rationale: weather and demand patterns repeat weekly.

Both read the stored price features, which already follow the Berlin wall clock
across clock changes and respect the cut-off.

Success criterion (proposal section 1):
  F1(LightGBM) >= max(F1(NaiveYesterday), F1(NaiveLastWeek)) + 0.10
"""

from __future__ import annotations


# TODO: define BaselineModel protocol with predict(df: pd.DataFrame) -> pd.Series

# TODO: implement class NaiveYesterday(BaselineModel) from price_h_d1

# TODO: implement class NaiveLastWeek(BaselineModel) from price_h_d7

# TODO: both classes predict 0, the majority class, when the lag is missing
