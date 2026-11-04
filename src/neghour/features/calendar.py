"""
Calendar, seasonality and holiday features.

All of these come from the Berlin wall clock. Negative prices follow three
cycles: time of day (solar peak at noon, no solar at night), day of week (lower
demand at weekends) and time of year (more solar in spring and summer). Sine
and cosine pairs let a model see that hour 23 is next to hour 0, and December
next to January.

Many German holidays are regional. `holiday_share` is the share of Germany's
population with a public holiday that day; `is_holiday` is 1 only when all 16
states are off. Luxembourg is about 0.8 % of the zone and is left out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from holidays import country_holidays

from neghour.availability import calendar_available_at
from neghour.config import BERLIN_TZ, STATE_POPULATION


def _cycle(values: pd.Index, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * np.asarray(values, dtype="float64") / period
    return np.sin(angle), np.cos(angle)


def _holiday_share(local: pd.DatetimeIndex) -> np.ndarray:
    days = pd.Series(local.date, index=range(len(local)))
    years = sorted({day.year for day in days.unique()})
    total = sum(STATE_POPULATION.values())
    shares = pd.Series(0.0, index=days.unique())
    for state, population in STATE_POPULATION.items():
        calendar = country_holidays("DE", subdiv=state, years=years)
        on_holiday = [day in calendar for day in shares.index]
        shares[on_holiday] += population / total
    return days.map(shares).round(6).to_numpy()


def add_calendar_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return the frame with calendar, seasonality and holiday columns added."""
    result = features.copy()
    hours = pd.DatetimeIndex(result["delivery_utc"])
    local = hours.tz_convert(BERLIN_TZ)

    result["hour_of_day"] = local.hour
    result["weekday"] = local.weekday
    result["month"] = local.month
    result["is_weekend"] = (local.weekday >= 5).astype("int64")

    result["hour_sin"], result["hour_cos"] = _cycle(local.hour, 24)
    result["weekday_sin"], result["weekday_cos"] = _cycle(local.weekday, 7)
    result["year_sin"], result["year_cos"] = _cycle(local.dayofyear, 365.25)

    result["holiday_share"] = _holiday_share(local)
    result["is_holiday"] = (result["holiday_share"] == 1.0).astype("int64")
    result["available_at_calendar"] = calendar_available_at(hours)
    return result
