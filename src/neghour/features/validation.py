"""
Schema and leakage validation. Runs before anything is written to storage.

A leakage violation raises LeakageError and aborts the run.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from neghour.timeutils import delivery_hours

FEATURE_COLUMNS: list[str] = [
    "delivery_utc",
    "delivery_day",
    "cutoff_utc",
    "available_at",
    # price features
    "price_h_d1",
    "price_h_d7",
    "neg_hours_last_7d",
    "available_at_price",
    "missing_price",
    # weather features
    "irradiance",
    "cloud_cover",
    "wind_100m",
    "temperature",
    "available_at_weather",
    "missing_weather",
    # actuals
    "load_d2",
    "solar_d2",
    "wind_d2",
    "load_h_d7",
    "available_at_actuals",
    "missing_actuals",
    "residual_load_proxy",
    # calendar
    "hour_of_day",
    "weekday",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
    "holiday_share",
    "is_holiday",
    "available_at_calendar",
]

LABEL_COLUMNS: list[str] = [
    "delivery_utc",
    "price_eur_mwh",
    "y_neg",
    "n_neg_quarters",
    "available_at",
]

AVAILABILITY_COLUMNS: list[str] = [
    "available_at",
    "available_at_price",
    "available_at_weather",
    "available_at_actuals",
    "available_at_calendar",
]


class LeakageError(ValueError):
    """Raised when a feature row contains data published after the cut-off."""


def validate_schema(frame: pd.DataFrame, expected: list[str]) -> None:
    """Raise ValueError if the frame's columns differ from the expected ones."""
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {missing}. Extra: {extra}.")


def validate_unique_hours(frame: pd.DataFrame) -> None:
    """Raise ValueError if any delivery hour appears twice."""
    duplicates = frame["delivery_utc"].duplicated().sum()
    if duplicates:
        raise ValueError(f"Duplicate delivery hours: {duplicates}.")


def validate_day_lengths(features: pd.DataFrame) -> None:
    """Raise ValueError unless each delivery day has all of its 23-25 hours."""
    for day, rows in features.groupby("delivery_day").size().items():
        expected = len(delivery_hours(date.fromisoformat(str(day))))
        if rows != expected:
            raise ValueError(f"{day} has {rows} rows, expected {expected}.")


def validate_no_leakage(features: pd.DataFrame) -> None:
    """Raise LeakageError if any availability timestamp is after the cut-off."""
    for column in AVAILABILITY_COLUMNS:
        late = features[features[column] > features["cutoff_utc"]]
        if not late.empty:
            first = late["delivery_utc"].min()
            raise LeakageError(
                f"{len(late)} rows have {column} after the cut-off, first at {first}."
            )
