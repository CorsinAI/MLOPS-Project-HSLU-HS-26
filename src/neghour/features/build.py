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

from neghour.timeutils import (
    cutoff_utc,
    delivery_days,
    delivery_hours,
    same_hour_earlier,
)

NEGATIVE_WINDOW_DAYS = 7


def build_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Return one label row per hour that has a published price."""
    labels = prices[["delivery_utc", "price_eur_mwh", "available_at"]].copy()
    labels["y_neg"] = (labels["price_eur_mwh"] < 0).astype("int64")
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


def build_features(
    prices: pd.DataFrame, first_day: date, last_day: date
) -> pd.DataFrame:
    """Return the full feature frame for the given delivery days."""
    features = build_price_features(prices, first_day, last_day)
    features["available_at"] = features["available_at_price"]
    return features
