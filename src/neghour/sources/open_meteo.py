"""
Open-Meteo weather forecast source.

For live pipeline runs: fetches the latest hourly forecast for the DE grid
centre coordinates defined in config.py.

For historical training rows: uses the Previous Runs API to reconstruct the
forecast that would have been available at 10:00 on D-1 (model run issued
roughly 2 days ahead), avoiding any look-ahead leakage.

Variables fetched (hourly):
  shortwave_radiation        → irradiance proxy
  cloud_cover                → cloud cover %
  wind_speed_100m            → wind at hub height
  temperature_2m             → temperature (residual-load proxy)

No API key required.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# TODO: implement fetch_forecast(delivery_date: date) -> pd.DataFrame
#       uses the live forecast endpoint; for daily pipeline runs
#       columns: delivery_date, delivery_hour, irradiance, cloud_cover,
#                wind_100m, temperature, available_at
# TODO: implement fetch_historical_forecast(delivery_date: date) -> pd.DataFrame
#       uses Previous Runs API with model_run = D-2 12:00 UTC
#       to reconstruct what was known at cut-off time on D-1
# TODO: map raw Open-Meteo variable names to canonical feature names
# TODO: derive residual_load_proxy = f(irradiance, wind_100m, temperature)
