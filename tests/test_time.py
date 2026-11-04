"""Tests for Berlin time rules and publication rules."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from neghour import availability
from neghour.timeutils import (
    berlin_today,
    cutoff_utc,
    delivery_days,
    delivery_hours,
    same_hour_earlier,
)


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 3, 29), 23),
        (date(2026, 6, 15), 24),
        (date(2026, 10, 25), 25),
    ],
)
def test_delivery_day_length_follows_dst(day: date, expected: int) -> None:
    assert len(delivery_hours(day)) == expected


def test_delivery_hours_are_utc_and_start_at_berlin_midnight() -> None:
    hours = delivery_hours(date(2026, 1, 15))
    assert str(hours.tz) == "UTC"
    assert hours[0] == pd.Timestamp("2026-01-14 23:00", tz="UTC")


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 1, 15), "2026-01-14 09:00"),
        (date(2026, 7, 15), "2026-07-14 08:00"),
    ],
)
def test_cutoff_is_ten_berlin_in_winter_and_summer(day: date, expected: str) -> None:
    assert cutoff_utc(day) == pd.Timestamp(expected, tz="UTC")


def test_berlin_today_differs_from_utc_late_in_the_evening() -> None:
    assert berlin_today(datetime(2026, 1, 14, 23, 30, tzinfo=UTC)) == date(2026, 1, 15)


def test_delivery_days_is_inclusive() -> None:
    assert delivery_days(date(2026, 1, 30), date(2026, 2, 1)) == [
        date(2026, 1, 30),
        date(2026, 1, 31),
        date(2026, 2, 1),
    ]


def test_same_hour_earlier_keeps_wall_clock_across_dst() -> None:
    hours = delivery_hours(date(2026, 3, 30))
    earlier = same_hour_earlier(hours, 2).tz_convert("Europe/Berlin")
    assert earlier[12].hour == 12
    assert earlier[12].day == 28


def test_same_hour_earlier_maps_skipped_hour_forward() -> None:
    two_oclock = pd.DatetimeIndex([pd.Timestamp("2026-03-30 00:00", tz="UTC")])
    earlier = same_hour_earlier(two_oclock, 1).tz_convert("Europe/Berlin")
    assert (earlier[0].day, earlier[0].hour) == (29, 3)


def test_same_hour_earlier_maps_repeated_hour_to_first_occurrence() -> None:
    hour = pd.DatetimeIndex([pd.Timestamp("2026-10-26 01:00", tz="UTC")])
    assert same_hour_earlier(hour, 1)[0] == pd.Timestamp("2026-10-25 00:00", tz="UTC")


def test_price_rule_is_thirteen_berlin_the_day_before() -> None:
    delivery = pd.DatetimeIndex([pd.Timestamp("2026-07-15 10:00", tz="UTC")])
    assert availability.price_available_at(delivery)[0] == pd.Timestamp(
        "2026-07-14 11:00", tz="UTC"
    )


def test_every_rule_clears_the_cutoff_for_its_latest_allowed_input() -> None:
    day = date(2026, 1, 15)
    cutoff = cutoff_utc(day)
    hours = delivery_hours(day)
    day_before = delivery_hours(date(2026, 1, 14))
    two_days_before = delivery_hours(date(2026, 1, 13))
    assert availability.price_available_at(day_before).max() <= cutoff
    assert availability.actuals_available_at(two_days_before).max() <= cutoff
    assert availability.weather_available_at(hours).max() <= cutoff
    assert availability.calendar_available_at(hours).max() <= cutoff


def test_price_of_the_delivery_day_is_blocked_at_the_cutoff() -> None:
    day = date(2026, 1, 15)
    assert availability.price_available_at(delivery_hours(day)).min() > cutoff_utc(day)
