"""
The 10:00 cut-off, enforced as tests (review point A3).

Checking `available_at` alone could pass by construction, because the builder
sets that column itself. So these tests also change or remove everything that
was published after the cut-off, and require the feature row to stay identical.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from neghour.features.build import build_features
from neghour.features.validation import (
    AVAILABILITY_COLUMNS,
    FEATURE_COLUMNS,
    LeakageError,
    validate_no_leakage,
)
from neghour.timeutils import cutoff_utc
from tests.frames import actuals_frame, price_frame, weather_frame

DAYS = [
    date(2026, 1, 15),
    date(2026, 7, 15),
    date(2026, 3, 29),
    date(2026, 3, 30),
    date(2026, 10, 25),
    date(2026, 10, 26),
]
HISTORY_DAYS = 8
FUTURE_DAYS = 2


def raw_frames(day: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return realistic raw frames that also hold data from after the cut-off."""
    first = day - timedelta(days=HISTORY_DAYS)
    last = day + timedelta(days=FUTURE_DAYS)
    random = np.random.default_rng(7)

    prices = price_frame(first, last)
    prices["price_eur_mwh"] = random.normal(40, 60, len(prices))

    weather = weather_frame(first, last)
    for column in ("irradiance", "cloud_cover", "wind_100m", "temperature"):
        weather[column] = random.uniform(1, 500, len(weather))

    actuals = actuals_frame(first, last)
    for column in ("load_mwh", "solar_mwh", "wind_mwh"):
        actuals[column] = random.uniform(1_000, 70_000, len(actuals))

    return prices, weather, actuals


def after_cutoff(frame: pd.DataFrame, column: str, day: date) -> pd.Series:
    return frame[column] > cutoff_utc(day)


@pytest.mark.parametrize("day", DAYS)
def test_no_feature_row_uses_data_after_its_cutoff(day: date) -> None:
    prices, weather, actuals = raw_frames(day)
    features = build_features(prices, weather, actuals, day, day)
    validate_no_leakage(features)
    assert (features["available_at"] <= cutoff_utc(day)).all()


@pytest.mark.parametrize("day", DAYS)
def test_rows_do_not_change_when_later_data_is_removed(day: date) -> None:
    prices, weather, actuals = raw_frames(day)
    full = build_features(prices, weather, actuals, day, day)
    truncated = build_features(
        prices[~after_cutoff(prices, "available_at", day)],
        weather[~after_cutoff(weather, "available_at_weather", day)],
        actuals[~after_cutoff(actuals, "available_at_actuals", day)],
        day,
        day,
    )
    pd.testing.assert_frame_equal(full, truncated)


@pytest.mark.parametrize("day", DAYS)
def test_rows_do_not_change_when_later_data_is_poisoned(day: date) -> None:
    prices, weather, actuals = raw_frames(day)
    full = build_features(prices, weather, actuals, day, day)

    prices.loc[after_cutoff(prices, "available_at", day), "price_eur_mwh"] = -9_999.0
    weather.loc[after_cutoff(weather, "available_at_weather", day), "wind_100m"] = 999.0
    actuals.loc[after_cutoff(actuals, "available_at_actuals", day), "load_mwh"] = -1.0

    poisoned = build_features(prices, weather, actuals, day, day)
    pd.testing.assert_frame_equal(full, poisoned)


def test_the_label_of_the_delivery_day_cannot_reach_the_row() -> None:
    day = date(2026, 7, 15)
    prices, weather, actuals = raw_frames(day)
    base = build_features(prices, weather, actuals, day, day)

    flipped = prices.copy()
    today = flipped["delivery_utc"] >= base["delivery_utc"].min()
    flipped.loc[today, "price_eur_mwh"] *= -1
    pd.testing.assert_frame_equal(
        base, build_features(flipped, weather, actuals, day, day)
    )


def test_every_availability_column_is_checked() -> None:
    declared = {name for name in FEATURE_COLUMNS if name.startswith("available_at")}
    assert declared == set(AVAILABILITY_COLUMNS)


@pytest.mark.parametrize("column", AVAILABILITY_COLUMNS)
def test_a_late_timestamp_in_any_column_is_rejected(column: str) -> None:
    day = date(2026, 1, 15)
    prices, weather, actuals = raw_frames(day)
    features = build_features(prices, weather, actuals, day, day)
    features.loc[3, column] = cutoff_utc(day) + pd.Timedelta(minutes=1)
    with pytest.raises(LeakageError):
        validate_no_leakage(features)
