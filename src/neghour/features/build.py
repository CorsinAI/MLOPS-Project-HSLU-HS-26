"""
Feature row assembly for the DE-LU negative-price prediction task.

Each feature row corresponds to one delivery hour h on day D.
ALL values must have been published before 10:00 on D-1 (the cut-off);
this is enforced via the `available_at` column before any row is returned.

Named features (proposal §3 — known by 10:00 on D-1):
  Weather  (Open-Meteo forecast issued ~D-2):
    irradiance, cloud_cover, wind_100m, temperature, residual_load_proxy
  Market   (SMARD, published by D-2 12:45 at the latest):
    price_h_d1, price_h_d7, neg_hours_last_7d, load_d2
  Lags     (SMARD actuals, published D-2):
    solar_d2, wind_d2
  Calendar:
    hour_of_day, weekday, is_holiday, month

Label (never used as a feature):
  y_neg = 1 if DE-LU day-ahead price for hour h on D < 0
  Published at D-1 12:45 — well after the cut-off.

Out-of-time split:
  train 2024 · validate H1 2025 · test Jul 2025–Jun 2026 · live from ~Nov 2026
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# TODO: implement build_row(delivery_date: date, delivery_hour: int) -> pd.Series
#       fetches all required source data and assembles one feature row
# TODO: implement build_day(delivery_date: date) -> pd.DataFrame
#       calls build_row for hours 0–23, returns 24-row DataFrame
# TODO: implement build_history(start: date, end: date) -> pd.DataFrame
#       loops over dates and concatenates; used for backfill
# TODO: enforce cut-off in all functions:
#       assert all(df["available_at"] < cutoff_dt), LeakageError
