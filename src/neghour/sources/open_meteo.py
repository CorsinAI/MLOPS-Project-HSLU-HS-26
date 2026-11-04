"""
Open-Meteo Previous Runs API. JSON, no key, CC BY 4.0, non-commercial free tier.

Backfill and live use this same request, so training and serving see the same
forecast vintage (review point A1). `_previous_day2` is the value predicted 48
hours before valid time, which is before the 10:00 cut-off on D-1 for every
hour of D. `_previous_day1` is not safe: for later hours of D it refers to a
forecast made after the cut-off.

Coverage: the default model has `_previous_day2` from 2024-01-20 12:00 UTC, and
`wind_speed_100m` from 2024-02-17 06:00 UTC. Earlier hours stay NaN.
"""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd

from neghour.availability import weather_available_at
from neghour.config import (
    OPEN_METEO_PREVIOUS_RUNS_URL,
    SOLAR_POINTS,
    TEMPERATURE_POINTS,
    WEATHER_POINTS,
    WIND_POINTS,
)
from neghour.sources.http import get_json, new_client
from neghour.timeutils import delivery_hours

API_VARIABLES = {
    "shortwave_radiation": "irradiance",
    "cloud_cover": "cloud_cover",
    "wind_speed_100m": "wind_100m",
    "temperature_2m": "temperature",
}
POINTS_FOR_FEATURE = {
    "irradiance": SOLAR_POINTS,
    "cloud_cover": SOLAR_POINTS,
    "wind_100m": WIND_POINTS,
    "temperature": TEMPERATURE_POINTS,
}
SUFFIX = "_previous_day2"
WEATHER_COLUMNS = list(POINTS_FOR_FEATURE)


def _point_frame(body: dict) -> pd.DataFrame:
    hourly = body["hourly"]
    frame = pd.DataFrame(
        {
            feature: pd.to_numeric(pd.Series(hourly[f"{name}{SUFFIX}"]))
            for name, feature in API_VARIABLES.items()
        }
    )
    frame.index = pd.to_datetime(hourly["time"], utc=True)
    return frame


def fetch_weather(
    first_day: date, last_day: date, client: httpx.Client | None = None
) -> pd.DataFrame:
    """Return aggregated `_previous_day2` forecasts per delivery hour."""
    http = client if client is not None else new_client()
    hours = delivery_hours(first_day, last_day)
    params = {
        "latitude": ",".join(str(lat) for lat, _ in WEATHER_POINTS.values()),
        "longitude": ",".join(str(lon) for _, lon in WEATHER_POINTS.values()),
        "hourly": ",".join(f"{name}{SUFFIX}" for name in API_VARIABLES),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "start_date": hours[0].date().isoformat(),
        "end_date": hours[-1].date().isoformat(),
    }
    bodies = get_json(http, OPEN_METEO_PREVIOUS_RUNS_URL, params)
    points = dict(zip(WEATHER_POINTS, map(_point_frame, bodies), strict=True))

    weather = pd.DataFrame(index=hours)
    for feature, names in POINTS_FOR_FEATURE.items():
        columns = pd.concat(
            [points[name][feature].reindex(hours) for name in names], axis=1
        )
        weather[feature] = columns.mean(axis=1, skipna=False)

    weather["available_at_weather"] = weather_available_at(hours)
    has_values = weather[WEATHER_COLUMNS].notna().any(axis=1)
    weather["available_at_weather"] = weather["available_at_weather"].where(has_values)
    return weather.reset_index(names="delivery_utc")


def empty_weather() -> pd.DataFrame:
    """Return a weather frame with no rows, used when Open-Meteo is unavailable."""
    no_times = pd.DatetimeIndex([], tz="UTC")
    columns = {name: pd.Series(dtype="float64") for name in WEATHER_COLUMNS}
    return pd.DataFrame(
        {"delivery_utc": no_times, **columns, "available_at_weather": no_times}
    )
