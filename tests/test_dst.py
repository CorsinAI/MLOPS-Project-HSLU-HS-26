"""Whole-pipeline tests for delivery days of 23 and 25 hours."""

from datetime import date, timedelta

import pandas as pd
import pytest

from neghour.config import BERLIN_TZ
from neghour.features.validation import validate_day_lengths, validate_unique_hours
from neghour.pipelines import feature_pipeline
from neghour.sources import open_meteo, smard
from neghour.storage.memory import MemoryStore
from tests.frames import actuals_frame, price_frame, weather_frame


@pytest.fixture(autouse=True)
def fake_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smard, "fetch_prices", price_frame)
    monkeypatch.setattr(
        smard,
        "fetch_negative_quarters",
        lambda first, last: smard.empty_negative_quarters(),
    )
    monkeypatch.setattr(smard, "fetch_actuals", actuals_frame)
    monkeypatch.setattr(open_meteo, "fetch_weather", weather_frame)


@pytest.mark.parametrize(
    ("day", "rows"),
    [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25), (date(2026, 10, 26), 24)],
)
def test_pipeline_writes_one_row_per_real_hour(day: date, rows: int) -> None:
    store = MemoryStore()
    feature_pipeline.run(day, day, store)
    assert len(store.read_features()) == rows
    assert len(store.read_labels()) == rows


def test_autumn_day_keys_stay_unique_though_a_local_hour_repeats() -> None:
    day = date(2026, 10, 25)
    store = MemoryStore()
    feature_pipeline.run(day, day, store)
    features = store.read_features()
    assert features["delivery_utc"].is_unique
    local_hours = pd.DatetimeIndex(features["delivery_utc"]).tz_convert(BERLIN_TZ).hour
    assert list(local_hours).count(2) == 2


def test_spring_day_has_no_local_two_oclock() -> None:
    day = date(2026, 3, 29)
    store = MemoryStore()
    feature_pipeline.run(day, day, store)
    hours = pd.DatetimeIndex(store.read_features()["delivery_utc"])
    assert 2 not in hours.tz_convert(BERLIN_TZ).hour


def test_lags_are_filled_on_the_days_after_each_clock_change() -> None:
    first = date(2026, 3, 29)
    store = MemoryStore()
    feature_pipeline.run(first, first + timedelta(days=1), store)
    features = store.read_features()
    assert features[["price_h_d1", "price_h_d7", "load_d2"]].notna().all().all()


def test_duplicate_hours_are_rejected() -> None:
    frame = price_frame(date(2026, 1, 15), date(2026, 1, 15))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_unique_hours(pd.concat([frame, frame.head(1)]))


def test_a_day_with_a_missing_hour_is_rejected() -> None:
    day = date(2026, 10, 25)
    frame = price_frame(day, day).assign(delivery_day=day.isoformat())
    with pytest.raises(ValueError, match="2026-10-25"):
        validate_day_lengths(frame.iloc[1:])
    validate_day_lengths(frame)
