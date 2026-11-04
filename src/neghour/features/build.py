"""
Feature and label assembly for the DE-LU negative-price task.

Every function here is pure: raw frames in, feature frames out. Live runs and
backfills use the same code, so they cannot drift apart.

Selection is by timestamp, never by when a job runs. Each delivery day D only
sees rows with available_at <= 10:00 Berlin on D-1, so a late run builds the
same row as an on-time run.

Label (never a feature):
  y_neg = 1 if the hourly mean DE-LU day-ahead price of the hour is < 0.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from neghour.config import (
    FULL_SUN_W_M2,
    SOLAR_CAPACITY_MW,
    WIND_CAPACITY_MW,
    WIND_RATED_SPEED_MS,
)
from neghour.features.calendar import add_calendar_features
from neghour.sources.open_meteo import WEATHER_COLUMNS
from neghour.timeutils import (
    cutoff_utc,
    delivery_days,
    delivery_hours,
    same_hour_earlier,
)

NEGATIVE_WINDOW_DAYS = 7
ACTUAL_COLUMNS = ["load_d2", "solar_d2", "wind_d2", "load_h_d7"]


def build_labels(prices: pd.DataFrame, negative_quarters: pd.Series) -> pd.DataFrame:
    """Return one label row per hour that has a published price.

    `n_neg_quarters` is stored so an "any quarter < 0" label can be tested later
    without rebuilding. It is NaN before the 15-minute switch.
    """
    labels = prices[["delivery_utc", "price_eur_mwh", "available_at"]].copy()
    labels["y_neg"] = (labels["price_eur_mwh"] < 0).astype("int64")
    labels["n_neg_quarters"] = negative_quarters.reindex(
        pd.DatetimeIndex(labels["delivery_utc"])
    ).to_numpy()
    return labels.sort_values("delivery_utc").reset_index(drop=True)


def _day_price_features(prices: pd.DataFrame, day: date) -> pd.DataFrame:
    cutoff = cutoff_utc(day)
    visible = prices[prices["available_at"] <= cutoff]
    price = visible.set_index("delivery_utc")["price_eur_mwh"]
    available = visible.set_index("delivery_utc")["available_at"]

    hours = delivery_hours(day)
    lag_1 = same_hour_earlier(hours, 1)
    lag_7 = same_hour_earlier(hours, 7)

    window = delivery_hours(
        day - timedelta(days=NEGATIVE_WINDOW_DAYS), day - timedelta(days=1)
    )
    window_prices = price.reindex(window)
    complete = window_prices.notna().all()
    negative_hours = float((window_prices < 0).sum()) if complete else float("nan")

    frame = pd.DataFrame(
        {
            "delivery_utc": hours,
            "delivery_day": day.isoformat(),
            "cutoff_utc": cutoff,
            "price_h_d1": price.reindex(lag_1).to_numpy(),
            "price_h_d7": price.reindex(lag_7).to_numpy(),
            "neg_hours_last_7d": negative_hours,
        }
    )
    frame["available_at_price"] = pd.concat(
        [
            available.reindex(lag_1).reset_index(drop=True),
            available.reindex(lag_7).reset_index(drop=True),
            pd.Series(available.reindex(window).max()).repeat(len(hours)).reset_index(
                drop=True
            ),
        ],
        axis=1,
    ).max(axis=1)
    frame["missing_price"] = (
        frame[["price_h_d1", "price_h_d7", "neg_hours_last_7d"]].isna().any(axis=1)
    )
    return frame


def build_price_features(
    prices: pd.DataFrame, first_day: date, last_day: date
) -> pd.DataFrame:
    """Return price features for every hour of the given delivery days."""
    frames = [
        _day_price_features(prices, day) for day in delivery_days(first_day, last_day)
    ]
    return pd.concat(frames, ignore_index=True)


def add_weather_features(features: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join weather forecasts, keeping only values published by the row cut-off."""
    merged = features.merge(weather, on="delivery_utc", how="left")
    too_late = ~(merged["available_at_weather"] <= merged["cutoff_utc"])
    merged[WEATHER_COLUMNS] = merged[WEATHER_COLUMNS].mask(too_late)
    merged["available_at_weather"] = merged["available_at_weather"].mask(too_late)
    merged["missing_weather"] = merged[WEATHER_COLUMNS].isna().any(axis=1)
    return merged


def add_actual_features(features: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Add D-2 actuals and the D-7 load, as published by each row cut-off."""
    result = features.copy()
    indexed = actuals.set_index("delivery_utc")
    hours = pd.DatetimeIndex(result["delivery_utc"])
    cutoff = result["cutoff_utc"]

    def visible(column: str, lags: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
        values = pd.Series(indexed[column].reindex(lags).to_numpy())
        available = pd.Series(indexed["available_at_actuals"].reindex(lags).array)
        return values.where(available <= cutoff), available.where(available <= cutoff)

    day_2 = same_hour_earlier(hours, 2)
    day_7 = same_hour_earlier(hours, 7)
    result["load_d2"], available_d2 = visible("load_mwh", day_2)
    result["solar_d2"], _ = visible("solar_mwh", day_2)
    result["wind_d2"], _ = visible("wind_mwh", day_2)
    result["load_h_d7"], available_d7 = visible("load_mwh", day_7)

    result["available_at_actuals"] = pd.concat(
        [available_d2, available_d7], axis=1
    ).max(axis=1)
    result["missing_actuals"] = result[ACTUAL_COLUMNS].isna().any(axis=1)
    return result


def add_residual_load_proxy(features: pd.DataFrame) -> pd.DataFrame:
    """Estimate demand left after wind and solar for each delivery hour.

    Expected demand is the load of the same hour one week earlier, which has the
    same weekday pattern. Wind and solar output come from the weather forecast
    and fixed capacities. Low or negative values mean a surplus of renewables.
    """
    result = features.copy()
    wind_share = (result["wind_100m"] / WIND_RATED_SPEED_MS).clip(upper=1.0) ** 3
    solar_share = (result["irradiance"] / FULL_SUN_W_M2).clip(upper=1.0)
    result["residual_load_proxy"] = (
        result["load_h_d7"]
        - WIND_CAPACITY_MW * wind_share
        - SOLAR_CAPACITY_MW * solar_share
    )
    return result


def build_features(
    prices: pd.DataFrame,
    weather: pd.DataFrame,
    actuals: pd.DataFrame,
    first_day: date,
    last_day: date,
) -> pd.DataFrame:
    """Return the full feature frame for the given delivery days."""
    features = build_price_features(prices, first_day, last_day)
    features = add_weather_features(features, weather)
    features = add_actual_features(features, actuals)
    features = add_residual_load_proxy(features)
    features = add_calendar_features(features)
    columns = [
        "available_at_price",
        "available_at_weather",
        "available_at_actuals",
        "available_at_calendar",
    ]
    for column in columns:
        features[column] = pd.to_datetime(features[column], utc=True)
    features["available_at"] = features[columns].max(axis=1)
    return features
