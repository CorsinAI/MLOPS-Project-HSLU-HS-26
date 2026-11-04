"""
SMARD (Bundesnetzagentur) data source. JSON, no key, CC BY 4.0.

Data comes in weekly chunks: an index file lists the chunk start times, and
each chunk holds [epoch_ms, value] pairs. There is no per-row publication
time, so `available_at` comes from the rules in neghour.availability.

The DE-LU day-ahead market clears 15-minute products since 01.10.2025. The
hourly series of filter 4169 stays continuous across that switch: before it
holds the hourly price, after it the mean of the four quarter hours, rounded
to two decimals. A mean between -0.005 and 0 therefore reads as 0.00.
"""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd

from neghour.availability import price_available_at
from neghour.config import SMARD_BASE_URL, SMARD_PRICE_FILTER, SMARD_REGION
from neghour.sources.http import get_json, new_client
from neghour.timeutils import delivery_hours

CHUNK_DAYS = 7


def _chunk_starts(
    client: httpx.Client, filter_id: int, resolution: str
) -> list[int]:
    url = f"{SMARD_BASE_URL}/{filter_id}/{SMARD_REGION}/index_{resolution}.json"
    return get_json(client, url)["timestamps"]


def fetch_series(
    filter_id: int,
    first_day: date,
    last_day: date,
    resolution: str = "hour",
    client: httpx.Client | None = None,
) -> pd.Series:
    """Return non-null values for Berlin days first_day to last_day, by UTC start."""
    http = client if client is not None else new_client()
    hours = delivery_hours(first_day, last_day)
    start = hours[0]
    stop = hours[-1] + pd.Timedelta(hours=1)
    chunk_span = pd.Timedelta(days=CHUNK_DAYS + 1)

    points: dict[int, float] = {}
    for chunk in _chunk_starts(http, filter_id, resolution):
        chunk_start = pd.Timestamp(chunk, unit="ms", tz="UTC")
        if chunk_start >= stop or chunk_start + chunk_span <= start:
            continue
        url = (
            f"{SMARD_BASE_URL}/{filter_id}/{SMARD_REGION}/"
            f"{filter_id}_{SMARD_REGION}_{resolution}_{chunk}.json"
        )
        for moment, value in get_json(http, url)["series"]:
            if value is not None:
                points[moment] = value

    series = pd.Series(points, dtype="float64").sort_index()
    series.index = pd.to_datetime(series.index, unit="ms", utc=True)
    series.index.name = "delivery_utc"
    return series[(series.index >= start) & (series.index < stop)]


def fetch_prices(
    first_day: date, last_day: date, client: httpx.Client | None = None
) -> pd.DataFrame:
    """Return hourly day-ahead prices with their rule-based available_at."""
    series = fetch_series(SMARD_PRICE_FILTER, first_day, last_day, client=client)
    index = pd.DatetimeIndex(series.index)
    return pd.DataFrame(
        {
            "delivery_utc": index,
            "price_eur_mwh": series.to_numpy(),
            "available_at": price_available_at(index),
        }
    )

