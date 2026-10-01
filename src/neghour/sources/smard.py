"""
SMARD (Bundesnetzagentur) data source.

Fetches DE-LU day-ahead electricity prices (filter 4169),
actual solar generation, wind generation, and load.
All data is available as JSON without an API key (CC BY 4.0).
Resolution: hourly, available since October 2018.

Every returned DataFrame includes an `available_at` column recording
the earliest timestamp at which SMARD published the data, so the
feature pipeline can enforce the 10:00 cut-off without leakage.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# TODO: implement fetch_prices(start: date, end: date) -> pd.DataFrame
#       columns: delivery_date, delivery_hour, price_eur_mwh, available_at
# TODO: implement fetch_generation(start: date, end: date) -> pd.DataFrame
#       columns: delivery_date, delivery_hour, solar_mwh, wind_mwh, available_at
# TODO: implement fetch_load(start: date, end: date) -> pd.DataFrame
#       columns: delivery_date, delivery_hour, load_mwh, available_at
# TODO: handle pagination (SMARD returns data in weekly chunks)
# TODO: raise SourceUnavailableError on HTTP errors so feature pipeline
#       can fall back to Energy-Charts automatically
