"""Tests for the feature pipeline, with sources faked and memory storage."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from neghour.features.validation import LeakageError
from neghour.pipelines import backfill, feature_pipeline
from neghour.sources import open_meteo, smard
from neghour.sources.http import SourceUnavailableError
from neghour.storage.memory import MemoryStore
from tests.frames import actuals_frame, price_frame, weather_frame

FIRST_DAY = date(2026, 1, 8)
LAST_DAY = date(2026, 1, 15)


def fake_quarters(first_day: date, last_day: date) -> pd.Series:
    return smard.empty_negative_quarters()


@pytest.fixture
def fake_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smard, "fetch_prices", price_frame)
    monkeypatch.setattr(smard, "fetch_negative_quarters", fake_quarters)
    monkeypatch.setattr(smard, "fetch_actuals", actuals_frame)
    monkeypatch.setattr(open_meteo, "fetch_weather", weather_frame)


def fail(*args: object) -> None:
    raise SourceUnavailableError("down")


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


@pytest.mark.usefixtures("fake_prices")
def test_weather_outage_gives_nan_and_keeps_good_stored_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore()
    feature_pipeline.run(FIRST_DAY, LAST_DAY, store)
    before = store.read_features()

    monkeypatch.setattr(open_meteo, "fetch_weather", fail)
    summary = feature_pipeline.run(FIRST_DAY, LAST_DAY, store, tolerate_missing=True)
    after = store.read_features()

    assert summary.failed_sources == ("open_meteo",)
    assert summary.feature_rows == 24
    newest = after["delivery_day"] == LAST_DAY.isoformat()
    assert after.loc[newest, "missing_weather"].all()
    assert after.loc[newest, "wind_100m"].isna().all()
    pd.testing.assert_frame_equal(
        after[~newest].reset_index(drop=True),
        before[before["delivery_day"] != LAST_DAY.isoformat()].reset_index(drop=True),
    )


@pytest.mark.usefixtures("fake_prices")
def test_smard_outage_keeps_stored_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryStore()
    feature_pipeline.run(FIRST_DAY, LAST_DAY, store)
    labels = store.read_labels()

    monkeypatch.setattr(smard, "fetch_prices", fail)
    feature_pipeline.run(FIRST_DAY, LAST_DAY, store, tolerate_missing=True)
    pd.testing.assert_frame_equal(labels, store.read_labels())


@pytest.mark.usefixtures("fake_prices")
def test_both_main_sources_down_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smard, "fetch_prices", fail)
    monkeypatch.setattr(open_meteo, "fetch_weather", fail)
    store = MemoryStore()
    with pytest.raises(SourceUnavailableError):
        feature_pipeline.run(FIRST_DAY, LAST_DAY, store, tolerate_missing=True)
    assert store.read_features().empty


@pytest.mark.usefixtures("fake_prices")
def test_strict_run_stops_on_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(open_meteo, "fetch_weather", fail)
    store = MemoryStore()
    with pytest.raises(SourceUnavailableError):
        feature_pipeline.run(FIRST_DAY, LAST_DAY, store)
    assert store.read_features().empty
