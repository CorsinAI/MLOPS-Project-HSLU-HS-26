"""Tests for SMARD, Open-Meteo, and Energy-Charts source modules."""

# TODO: test SMARD fetch_prices returns DataFrame with correct columns and dtypes
# TODO: test SMARD fetch_prices available_at is always <= price publication time
# TODO: test SMARD raises SourceUnavailableError on non-200 HTTP response
# TODO: test Open-Meteo fetch_historical_forecast uses model_run = D-2 12:00 UTC
# TODO: test Open-Meteo canonical column names match EXPECTED_COLUMNS in validation
# TODO: test Energy-Charts fetch_generation resamples 15-min data to hourly correctly
# TODO: test Energy-Charts is_fallback=True flag is present in all returned rows
