"""
Publication rules: when each kind of data counts as available.

SMARD has no per-row publication time, so `available_at` is rule-based. Every
rule is deliberately conservative and lives here, in one place. A feature row
for delivery day D may only use data with available_at <= 10:00 Berlin on D-1.
"""

from __future__ import annotations

import pandas as pd

from neghour.config import BERLIN_TZ

PRICE_PUBLICATION_HOUR = 13
ACTUALS_LAG = pd.Timedelta(hours=6)
WEATHER_LEAD = pd.Timedelta(hours=48)
WEATHER_MARGIN = pd.Timedelta(hours=6)


def _day_before_at(hours: pd.DatetimeIndex, hour: int) -> pd.DatetimeIndex:
    """Return the given Berlin hour on the day before each timestamp, in UTC."""
    midnight = hours.tz_convert(BERLIN_TZ).normalize().tz_localize(None)
    local = midnight - pd.Timedelta(days=1) + pd.Timedelta(hours=hour)
    return local.tz_localize(BERLIN_TZ).tz_convert("UTC")


def price_available_at(delivery: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Day-ahead prices for day X count as published at 13:00 Berlin on X-1."""
    return _day_before_at(delivery, PRICE_PUBLICATION_HOUR)


def actuals_available_at(hour_start: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """SMARD actuals count as published 6 hours after the hour ends."""
    return hour_start + pd.Timedelta(hours=1) + ACTUALS_LAG


def weather_available_at(valid_time: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """A `_previous_day2` value counts as published 42 hours before valid time."""
    return valid_time - WEATHER_LEAD + WEATHER_MARGIN


def calendar_available_at(delivery: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Calendar facts are known in advance; they count from 00:00 Berlin on D-1."""
    return _day_before_at(delivery, 0)
