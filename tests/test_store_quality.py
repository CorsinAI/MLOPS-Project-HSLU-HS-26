"""The stored-data report must catch each fault it claims to check."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from neghour.features.build import build_features, build_labels
from neghour.sources import smard
from neghour.timeutils import cutoff_utc
from scripts.check_store_quality import (
    CLOCK_CHANGE_DAYS,
    WIND_START,
    run_checks,
)
from tests.frames import actuals_frame, price_frame, weather_frame

FIRST = date(2024, 3, 29)
LAST = date(2024, 4, 2)


def good_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = price_frame(date(2024, 3, 15), LAST)
    weather = weather_frame(FIRST, LAST)
    actuals = actuals_frame(date(2024, 3, 15), LAST)
    features = build_features(prices, weather, actuals, FIRST, LAST)
    window = prices["delivery_utc"].isin(features["delivery_utc"])
    return features, build_labels(prices[window], smard.empty_negative_quarters())


def failures_of(features: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    # The window here holds only one clock-change day, so ignore the others.
    return [
        failure
        for failure in run_checks(features, labels)
        if not (failure[:10] in CLOCK_CHANGE_DAYS and failure[:10] != "2024-03-31")
    ]


def test_good_tables_pass() -> None:
    assert failures_of(*good_tables()) == []


def duplicate_a_row(features: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([features, features.iloc[[5]]], ignore_index=True)


def drop_the_clock_change_day_hour(features: pd.DataFrame) -> pd.DataFrame:
    return features.drop(features.index[features["delivery_day"] == "2024-03-31"][3])


def leak_one_row(features: pd.DataFrame) -> pd.DataFrame:
    leaked = features.copy()
    late = cutoff_utc(date(2024, 3, 31)) + pd.Timedelta(minutes=1)
    leaked.loc[leaked["delivery_day"] == "2024-03-31", "available_at_price"] = late
    return leaked


def shift_the_stored_cutoff(features: pd.DataFrame) -> pd.DataFrame:
    return features.assign(cutoff_utc=features["cutoff_utc"] + pd.Timedelta(hours=1))


def empty_wind_after_its_start(features: pd.DataFrame) -> pd.DataFrame:
    assert features["delivery_utc"].max() > WIND_START
    return features.assign(wind_100m=np.nan)


def empty_a_price_feature(features: pd.DataFrame) -> pd.DataFrame:
    damaged = features.copy()
    damaged.loc[damaged.index[0], "price_h_d1"] = np.nan
    return damaged


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        (duplicate_a_row, "duplicate UTC hours"),
        (drop_the_clock_change_day_hour, "2024-03-31 has 22 rows"),
        (leak_one_row, "available_at_price after the cut-off"),
        (shift_the_stored_cutoff, "stored cutoff_utc differs"),
        (empty_wind_after_its_start, "wind_100m is empty"),
        (empty_a_price_feature, "price_h_d1 is empty"),
    ],
)
def test_each_fault_is_reported(
    damage: Callable[[pd.DataFrame], pd.DataFrame], expected: str
) -> None:
    features, labels = good_tables()
    assert any(expected in f for f in failures_of(damage(features), labels))


def test_labels_for_other_hours_are_reported() -> None:
    features, labels = good_tables()
    failures = failures_of(features, labels.iloc[1:])
    assert "features and labels cover different hours" in failures


def test_times_returned_without_a_timezone_are_read_as_utc() -> None:
    features, labels = good_tables()
    stored = features.assign(
        delivery_utc=features["delivery_utc"].dt.tz_localize(None),
        cutoff_utc=features["cutoff_utc"].dt.tz_localize(None),
    )
    assert failures_of(stored, labels) == []
