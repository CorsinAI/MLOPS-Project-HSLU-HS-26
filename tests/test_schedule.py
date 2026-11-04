"""The cron slots must keep their purpose in both CET and CEST."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from neghour import config

WORKFLOW = Path(".github/workflows/daily_pipeline.yml")
WINTER = datetime(2026, 1, 15, tzinfo=UTC)
SUMMER = datetime(2026, 7, 15, tzinfo=UTC)
DAY_AHEAD_PUBLISHED_HOUR = 13


def slots() -> list[tuple[int, int]]:
    """Return (hour, minute) in UTC for each cron line."""
    found = re.findall(r'cron: "(\d+) (\d+) \* \* \*"', WORKFLOW.read_text("utf-8"))
    return [(int(hour), int(minute)) for minute, hour in found]


def berlin_time(hour: int, minute: int, day: datetime) -> tuple[int, int]:
    local = day.replace(hour=hour, minute=minute).astimezone(config.BERLIN_TZ)
    return local.hour, local.minute


def test_there_are_two_morning_slots_and_one_afternoon_slot() -> None:
    assert sorted(slots()) == [(1, 17), (5, 17), (12, 17)]


@pytest.mark.parametrize("day", [WINTER, SUMMER])
def test_morning_slots_start_before_the_cut_off(day: datetime) -> None:
    for hour, minute in slots()[:2]:
        assert berlin_time(hour, minute, day)[0] < config.CUTOFF_HOUR


@pytest.mark.parametrize("day", [WINTER, SUMMER])
def test_the_label_slot_starts_after_day_ahead_publication(day: datetime) -> None:
    hour, minute = slots()[2]
    assert berlin_time(hour, minute, day) >= (DAY_AHEAD_PUBLISHED_HOUR, 0)
