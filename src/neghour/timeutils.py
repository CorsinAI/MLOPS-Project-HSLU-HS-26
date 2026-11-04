"""
Time rules for NegativeHours.

All stored timestamps are UTC. Delivery days and the cut-off are defined in
Europe/Berlin, so a delivery day has 23, 24 or 25 hours. "Today" always comes
from the Berlin calendar, never from the runner clock.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from neghour.config import BERLIN_TZ, CUTOFF_HOUR


def berlin_today(now: datetime | None = None) -> date:
    """Return the current calendar day in Berlin."""
    moment = now if now is not None else datetime.now(UTC)
    return moment.astimezone(BERLIN_TZ).date()


def _berlin_midnight(day: date) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(BERLIN_TZ)


def cutoff_utc(delivery_day: date) -> pd.Timestamp:
    """Return the data cut-off for a delivery day: 10:00 Berlin on D-1, in UTC."""
    local = pd.Timestamp(delivery_day - timedelta(days=1)) + pd.Timedelta(
        hours=CUTOFF_HOUR
    )
    return local.tz_localize(BERLIN_TZ).tz_convert("UTC")


def delivery_hours(first_day: date, last_day: date | None = None) -> pd.DatetimeIndex:
    """Return the UTC start of every hour from first_day to last_day, inclusive."""
    end_day = last_day if last_day is not None else first_day
    start = _berlin_midnight(first_day).tz_convert("UTC")
    stop = _berlin_midnight(end_day + timedelta(days=1)).tz_convert("UTC")
    return pd.date_range(start, stop, freq="h", inclusive="left", name="delivery_utc")


def delivery_days(first_day: date, last_day: date) -> list[date]:
    """Return every calendar day from first_day to last_day, inclusive."""
    return [
        first_day + timedelta(days=offset)
        for offset in range((last_day - first_day).days + 1)
    ]


def same_hour_earlier(hours: pd.DatetimeIndex, days: int) -> pd.DatetimeIndex:
    """Return the same Berlin wall-clock hour a number of days earlier, in UTC.

    A wall-clock hour skipped by the spring change maps to the next valid hour.
    A wall-clock hour repeated by the autumn change maps to its first occurrence.
    """
    wall_clock = hours.tz_convert(BERLIN_TZ).tz_localize(None) - pd.Timedelta(days=days)
    first_occurrence = np.ones(len(wall_clock), dtype=bool)
    localized = wall_clock.tz_localize(
        BERLIN_TZ, nonexistent="shift_forward", ambiguous=first_occurrence
    )
    return localized.tz_convert("UTC")
