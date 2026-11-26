"""Inference pipeline: timing guard, frozen rows, late runs, stored columns."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neghour.models.lgbm_model import MODEL_FEATURES, NegHourLGBM
from neghour.models.registry import LocalRegistry, NoModelError
from neghour.pipelines import inference_pipeline as ip
from neghour.sources import open_meteo, smard
from neghour.storage.memory import MemoryStore
from tests.frames import actuals_frame, price_frame, weather_frame
from tests.test_models import synthetic_rows

DAY = date(2026, 1, 16)
EARLY = datetime(2026, 1, 15, 8, 30, tzinfo=UTC)  # 09:30 Berlin, before cut-off
ON_TIME = datetime(2026, 1, 15, 9, 5, tzinfo=UTC)  # 10:05 Berlin
LATE = datetime(2026, 1, 15, 11, 30, tzinfo=UTC)  # 12:30 Berlin
DST_DAY = date(2026, 3, 29)  # 23 hours


@pytest.fixture(autouse=True)
def fake_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smard, "fetch_prices", price_frame)
    monkeypatch.setattr(smard, "fetch_negative_quarters",
                        lambda a, b: smard.empty_negative_quarters())
    monkeypatch.setattr(smard, "fetch_actuals", actuals_frame)
    monkeypatch.setattr(open_meteo, "fetch_weather", weather_frame)


@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory) -> LocalRegistry:
    X_train, y_train = synthetic_rows(2000, seed=1)
    X_val, y_val = synthetic_rows(800, seed=2)
    model = NegHourLGBM(params={"n_estimators": 50}).fit(X_train, y_train, X_val, y_val)
    registry = LocalRegistry(tmp_path_factory.mktemp("models"))
    registry.register(model, {"f1": 0.7}, "test model")
    return registry


def test_a_run_before_the_cut_off_is_refused(registry: LocalRegistry) -> None:
    store = MemoryStore()
    with pytest.raises(ip.TooEarlyError):
        ip.run(DAY, store, registry, now=EARLY)
    assert store.read_predictions().empty and store.read_features().empty


def test_an_on_time_run_refreshes_features_and_stores_every_hour(
    registry: LocalRegistry,
) -> None:
    store = MemoryStore()
    summary, predictions = ip.run(DAY, store, registry, now=ON_TIME)
    stored = store.read_predictions()
    assert list(stored.columns) == ip.PREDICTION_COLUMNS
    assert summary.rows == len(stored) == 24
    assert summary.on_time and summary.written and summary.model_version == 1
    assert stored["on_time"].all()
    assert ((stored["p_neg"] >= 0) & (stored["p_neg"] <= 1)).all()
    assert set(stored["y_pred"]) <= {0, 1}
    assert (stored["predicted_at"] == pd.Timestamp(ON_TIME)).all()
    assert len(store.read_features()) == 24


def test_a_clock_change_day_gets_23_predictions(registry: LocalRegistry) -> None:
    store = MemoryStore()
    now = datetime(2026, 3, 28, 9, 30, tzinfo=UTC)
    summary, _ = ip.run(DST_DAY, store, registry, now=now)
    assert summary.rows == 23


def test_the_default_day_is_berlin_tomorrow(registry: LocalRegistry) -> None:
    summary, _ = ip.run(None, MemoryStore(), registry, now=ON_TIME)
    assert summary.delivery_day == DAY


def test_a_late_run_is_flagged_and_keeps_the_on_time_prediction(
    registry: LocalRegistry,
) -> None:
    store = MemoryStore()
    ip.run(DAY, store, registry, now=ON_TIME)
    before = store.read_predictions()
    summary, _ = ip.run(DAY, store, registry, now=LATE)
    assert not summary.on_time and not summary.written
    assert "on-time prediction is kept" in summary.note
    pd.testing.assert_frame_equal(store.read_predictions(), before)


def test_a_late_run_without_an_earlier_prediction_is_stored_as_late(
    registry: LocalRegistry,
) -> None:
    store = MemoryStore()
    summary, _ = ip.run(DAY, store, registry, now=LATE)
    assert summary.written
    assert not store.read_predictions()["on_time"].any()


def test_a_late_run_does_not_rewrite_frozen_feature_rows(
    registry: LocalRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MemoryStore()
    ip.run(DAY, store, registry, now=ON_TIME)
    features = store.read_features()
    monkeypatch.setattr(open_meteo, "fetch_weather",
                        lambda a, b: weather_frame(a, b).assign(wind_100m=99.0))
    ip.run(DAY, store, registry, now=LATE)
    pd.testing.assert_frame_equal(store.read_features(), features)


def test_a_source_outage_uses_the_stored_rows(
    registry: LocalRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neghour.sources.http import SourceUnavailableError

    store = MemoryStore()
    ip.run(DAY, store, registry, now=ON_TIME, write=False)

    def down(*args: object) -> None:
        raise SourceUnavailableError("down")

    monkeypatch.setattr(smard, "fetch_prices", down)
    monkeypatch.setattr(open_meteo, "fetch_weather", down)
    summary, _ = ip.run(DAY, store, registry, now=ON_TIME)
    assert summary.written and summary.rows == 24


def test_a_dry_run_writes_no_prediction(registry: LocalRegistry) -> None:
    store = MemoryStore()
    summary, predictions = ip.run(DAY, store, registry, now=ON_TIME, write=False)
    assert not summary.written and len(predictions) == 24
    assert store.read_predictions().empty


def test_missing_inputs_are_marked(
    registry: LocalRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neghour.sources.http import SourceUnavailableError

    def down(*args: object) -> None:
        raise SourceUnavailableError("down")

    monkeypatch.setattr(open_meteo, "fetch_weather", down)
    summary, predictions = ip.run(DAY, MemoryStore(), registry, now=ON_TIME)
    assert predictions["missing_inputs"].all()
    assert summary.rows_with_missing_inputs == 24


def test_no_registered_model_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(NoModelError):
        ip.run(DAY, MemoryStore(), LocalRegistry(tmp_path), now=ON_TIME)


def test_an_incomplete_stored_day_is_refused(registry: LocalRegistry) -> None:
    store = MemoryStore()
    ip.run(DAY, store, registry, now=ON_TIME, write=False)
    store._features = store._features.iloc[:-1]
    with pytest.raises(ValueError, match="has 23 rows, expected 24"):
        ip.run(DAY, store, registry, now=ON_TIME, refresh=False)


def test_predictions_use_only_model_features(registry: LocalRegistry) -> None:
    model, _ = registry.load_latest()
    assert model.features == MODEL_FEATURES
    rows = pd.DataFrame(np.zeros((2, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    assert model.predict_proba(rows.assign(fetched_at=1.0)).shape == (2,)
