"""Small raw frames shared by the tests."""

from datetime import date

import pandas as pd

from neghour.availability import (
    actuals_available_at,
    price_available_at,
    weather_available_at,
)
from neghour.timeutils import delivery_hours


def price_frame(first_day: date, last_day: date, value: float = 50.0) -> pd.DataFrame:
    hours = delivery_hours(first_day, last_day)
    return pd.DataFrame(
        {
            "delivery_utc": hours,
            "price_eur_mwh": value,
            "available_at": price_available_at(hours),
        }
    )


def weather_frame(first_day: date, last_day: date) -> pd.DataFrame:
    hours = delivery_hours(first_day, last_day)
    return pd.DataFrame(
        {
            "delivery_utc": hours,
            "irradiance": 500.0,
            "cloud_cover": 50.0,
            "wind_100m": 6.0,
            "temperature": 12.0,
            "available_at_weather": weather_available_at(hours),
        }
    )


def actuals_frame(first_day: date, last_day: date) -> pd.DataFrame:
    hours = delivery_hours(first_day, last_day)
    return pd.DataFrame(
        {
            "delivery_utc": hours,
            "load_mwh": 60_000.0,
            "solar_mwh": 5_000.0,
            "wind_mwh": 10_000.0,
            "available_at_actuals": actuals_available_at(hours),
        }
    )
