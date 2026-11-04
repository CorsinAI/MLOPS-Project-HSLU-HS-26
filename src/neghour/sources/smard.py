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

from neghour.availability import actuals_available_at, price_available_at
from neghour.config import (
    SMARD_BASE_URL,
    SMARD_LOAD_FILTER,
    SMARD_PRICE_FILTER,
    SMARD_REGION,
    SMARD_SOLAR_FILTER,
    SMARD_WIND_OFFSHORE_FILTER,
    SMARD_WIND_ONSHORE_FILTER,
)
from neghour.sources.http import get_json, new_client
from neghour.timeutils import delivery_hours

CHUNK_DAYS = 7
QUARTER_SWITCH_DAY = date(2025, 10, 1)


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



def fetch_negative_quarters(
    first_day: date, last_day: date, client: httpx.Client | None = None
) -> pd.Series:
    """Return how many quarter hours of each hour were negative.

    Only hours from the 15-minute switch on have four separate quarter prices.
    """
    start = max(first_day, QUARTER_SWITCH_DAY)
    if start > last_day:
        return empty_negative_quarters()
    quarters = fetch_series(
        SMARD_PRICE_FILTER, start, last_day, resolution="quarterhour", client=client
    )
    hour = quarters.index.floor("h")
    counts = (quarters < 0).groupby(hour).sum()
    complete = quarters.groupby(hour).size() == 4
    return counts[complete].astype("float64").rename("n_neg_quarters")


def fetch_actuals(
    first_day: date, last_day: date, client: httpx.Client | None = None
) -> pd.DataFrame:
    """Return hourly load, solar and wind with their rule-based available_at."""
    http = client if client is not None else new_client()

    def series(filter_id: int) -> pd.Series:
        return fetch_series(filter_id, first_day, last_day, client=http)

    raw = pd.DataFrame(
        {
            "load_mwh": series(SMARD_LOAD_FILTER),
            "solar_mwh": series(SMARD_SOLAR_FILTER),
            "onshore": series(SMARD_WIND_ONSHORE_FILTER),
            "offshore": series(SMARD_WIND_OFFSHORE_FILTER),
        }
    )
    raw["wind_mwh"] = raw["onshore"] + raw["offshore"]
    index = pd.DatetimeIndex(raw.index)
    return pd.DataFrame(
        {
            "delivery_utc": index,
            "load_mwh": raw["load_mwh"].to_numpy(),
            "solar_mwh": raw["solar_mwh"].to_numpy(),
            "wind_mwh": raw["wind_mwh"].to_numpy(),
            "available_at_actuals": actuals_available_at(index),
        }
    )


def _no_times() -> pd.DatetimeIndex:
    return pd.DatetimeIndex([], tz="UTC")


def empty_prices() -> pd.DataFrame:
    """Return a price frame with no rows, used when SMARD is unavailable."""
    return pd.DataFrame(
        {
            "delivery_utc": _no_times(),
            "price_eur_mwh": pd.Series(dtype="float64"),
            "available_at": _no_times(),
        }
    )


def empty_actuals() -> pd.DataFrame:
    """Return an actuals frame with no rows, used when SMARD is unavailable."""
    return pd.DataFrame(
        {
            "delivery_utc": _no_times(),
            "load_mwh": pd.Series(dtype="float64"),
            "solar_mwh": pd.Series(dtype="float64"),
            "wind_mwh": pd.Series(dtype="float64"),
            "available_at_actuals": _no_times(),
        }
    )


def empty_negative_quarters() -> pd.Series:
    """Return an empty quarter count series."""
    return pd.Series(dtype="float64", index=_no_times(), name="n_neg_quarters")
