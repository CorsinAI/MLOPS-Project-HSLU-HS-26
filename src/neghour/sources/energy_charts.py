"""
Energy-Charts fallback data source (Fraunhofer ISE).

Used when the SMARD API is unavailable. Provides DE-LU day-ahead prices
and generation mix via the public JSON API (no key required).
Native resolution: 15 minutes — resampled to hourly to match SMARD format.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# TODO: implement fetch_prices(start: date, end: date) -> pd.DataFrame
#       columns: delivery_date, delivery_hour, price_eur_mwh,
#                available_at, is_fallback=True
# TODO: implement fetch_generation(start: date, end: date) -> pd.DataFrame
#       columns: delivery_date, delivery_hour, solar_mwh, wind_mwh,
#                available_at, is_fallback=True
# TODO: resample 15-min intervals to hourly means/sums before returning
# TODO: raise SourceUnavailableError on HTTP errors (no further fallback)
