"""Tests for calendar, seasonality and holiday features."""

from datetime import date

import pandas as pd
import pytest

from neghour.config import STATE_POPULATION
from neghour.features.calendar import add_calendar_features
from neghour.timeutils import delivery_hours


def calendar_for(day: date) -> pd.DataFrame:
    frame = pd.DataFrame({"delivery_utc": delivery_hours(day)})
    return add_calendar_features(frame)


def population_share(*states: str) -> float:
    total = sum(STATE_POPULATION.values())
    return sum(STATE_POPULATION[state] for state in states) / total


def test_state_populations_sum_to_the_published_total() -> None:
    assert sum(STATE_POPULATION.values()) == 83_467_117


def test_calendar_uses_berlin_wall_clock() -> None:
    features = calendar_for(date(2026, 1, 15))
    assert features["hour_of_day"].tolist() == list(range(24))
    assert (features["weekday"] == 3).all()
    assert (features["month"] == 1).all()


@pytest.mark.parametrize(
    ("day", "rows", "weekend"),
    [
        (date(2026, 3, 29), 23, 1),
        (date(2026, 10, 25), 25, 1),
        (date(2026, 6, 15), 24, 0),
    ],
)
def test_calendar_has_one_row_per_hour_on_every_day_length(
    day: date, rows: int, weekend: int
) -> None:
    features = calendar_for(day)
    assert len(features) == rows
    assert (features["is_weekend"] == weekend).all()


def test_hour_cycle_places_midnight_next_to_late_evening() -> None:
    features = calendar_for(date(2026, 1, 15))
    gap = (features["hour_sin"].iloc[23] - features["hour_sin"].iloc[0]) ** 2 + (
        features["hour_cos"].iloc[23] - features["hour_cos"].iloc[0]
    ) ** 2
    assert gap < 0.1


def test_national_holiday_is_a_full_share() -> None:
    features = calendar_for(date(2026, 12, 25))
    assert (features["holiday_share"] == 1.0).all()
    assert (features["is_holiday"] == 1).all()


def test_normal_day_has_no_share() -> None:
    features = calendar_for(date(2026, 6, 16))
    assert (features["holiday_share"] == 0.0).all()
    assert (features["is_holiday"] == 0).all()


def test_regional_holiday_gives_a_partial_share_without_the_flag() -> None:
    features = calendar_for(date(2026, 1, 6))
    expected = population_share("BW", "BY", "ST")
    assert features["holiday_share"].iloc[0] == pytest.approx(expected, abs=1e-6)
    assert (features["is_holiday"] == 0).all()


def test_reformation_day_covers_its_nine_states() -> None:
    features = calendar_for(date(2026, 10, 31))
    expected = population_share(
        "BB", "HB", "HH", "MV", "NI", "SH", "SN", "ST", "TH"
    )
    assert features["holiday_share"].iloc[0] == pytest.approx(expected, abs=1e-6)
