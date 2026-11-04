"""Tests for the feature pipeline, with sources faked and memory storage."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from neghour.availability import price_available_at, weather_available_at
from neghour.features.validation import LeakageError
from neghour.pipelines import backfill, feature_pipeline
from neghour.sources import open_meteo, smard
from neghour.storage.memory import MemoryStore
from neghour.timeutils import delivery_hours

FIRST_DAY = date(2026, 1, 8)
LAST_DAY = date(2026, 1, 15)


@pytest.fixture
def fake_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    def fetch(first_day: date, last_day: date) -> pd.DataFrame:
        hours = delivery_hours(first_day, last_day)
        return pd.DataFrame(
            {
                "delivery_utc": hours,
                "price_eur_mwh": 50.0,
                "available_at": price_available_at(hours),
            }
        )

    def fetch_weather(first_day: date, last_day: date) -> pd.DataFrame:
        hours = delivery_hours(first_day, last_day)
        return pd.DataFrame(
            {
                "delivery_utc": hours,
                "irradiance": 100.0,
                "cloud_cover": 50.0,
                "wind_100m": 8.0,
                "temperature": 12.0,
                "available_at_weather": weather_available_at(hours),
            }
        )

    monkeypatch.setattr(smard, "fetch_prices", fetch)
    monkeypatch.setattr(open_meteo, "fetch_weather", fetch_weather)


def test_daily_window_uses_the_berlin_calendar() -> None:
    late_evening_utc = datetime(2026, 1, 14, 23, 30, tzinfo=UTC)
    assert feature_pipeline.daily_window(late_evening_utc) == (
        date(2026, 1, 8),
        date(2026, 1, 16),
    )


@pytest.mark.usefixtures("fake_prices")
def test_run_writes_features_and_labels_for_the_window() -> None:
    store = MemoryStore()
    summary = feature_pipeline.run(FIRST_DAY, LAST_DAY, store)
    hours = 24 * 8
    assert summary.feature_rows == hours
    assert summary.label_rows == hours
    assert len(store.read_features()) == hours
    assert store.read_labels()["delivery_utc"].is_unique


@pytest.mark.usefixtures("fake_prices")
def test_second_run_adds_no_rows() -> None:
    store = MemoryStore()
    feature_pipeline.run(FIRST_DAY, LAST_DAY, store)
    first = store.read_features()
    feature_pipeline.run(FIRST_DAY, LAST_DAY, store)
    pd.testing.assert_frame_equal(first, store.read_features())


@pytest.mark.usefixtures("fake_prices")
def test_run_writes_nothing_on_leakage(monkeypatch: pytest.MonkeyPatch) -> None:
    original = feature_pipeline.build_features

    def leaky(*args: object) -> pd.DataFrame:
        frame = original(*args)
        return frame.assign(available_at=frame["cutoff_utc"] + timedelta(seconds=1))

    monkeypatch.setattr(feature_pipeline, "build_features", leaky)
    store = MemoryStore()
    with pytest.raises(LeakageError):
        feature_pipeline.run(FIRST_DAY, LAST_DAY, store)
    assert store.read_features().empty


def test_backfill_arguments() -> None:
    assert backfill.parse_args([]).first_day == date(2024, 1, 1)
    argv = ["--first-day", "2025-02-03", "--last-day", "2025-02-04"]
    args = backfill.parse_args(argv)
    assert (args.first_day, args.last_day) == (date(2025, 2, 3), date(2025, 2, 4))
