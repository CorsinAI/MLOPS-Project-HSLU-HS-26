"""Training pipeline on a synthetic stored dataset with a planted signal."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neghour import config
from neghour.models.lgbm_model import MODEL_FEATURES
from neghour.models.registry import LocalRegistry, NoModelError
from neghour.pipelines import training_pipeline as tp
from neghour.storage.memory import MemoryStore
from neghour.timeutils import delivery_hours

FAST = {"n_estimators": 80}


def synthetic_store(first: date, last: date, seed: int = 0) -> MemoryStore:
    """Store rows where low residual load at midday makes a negative hour."""
    rng = np.random.default_rng(seed)
    hours = delivery_hours(first, last)
    local = hours.tz_convert(config.BERLIN_TZ)
    n = len(hours)
    features = pd.DataFrame(rng.normal(size=(n, len(MODEL_FEATURES))),
                            columns=MODEL_FEATURES)
    features["delivery_utc"] = hours
    features["delivery_day"] = [d.isoformat() for d in local.date]
    features["hour_of_day"] = local.hour
    features["month"] = local.month
    features["is_weekend"] = (local.weekday >= 5).astype("int64")
    midday = features["hour_of_day"].between(10, 15)
    signal = -2.0 * features["residual_load_proxy"] + 1.5 * midday
    y = (signal + rng.normal(scale=0.4, size=n) > 3.0).astype("int64")
    # Yesterday's price carries part of the signal, so the baseline is not empty.
    features["price_h_d1"] = np.where(
        (y == 1) & (rng.random(n) < 0.4), -5.0, 40.0
    )
    labels = pd.DataFrame({"delivery_utc": hours, "y_neg": y,
                           "price_eur_mwh": np.where(y == 1, -5.0, 50.0)})
    store = MemoryStore()
    store.upsert_features(features)
    store.upsert_labels(labels)
    return store


@pytest.fixture(scope="module")
def store() -> MemoryStore:
    return synthetic_store(date(2024, 1, 1), date(2026, 8, 31))


def test_dataset_starts_at_the_train_start_and_needs_a_label() -> None:
    source = synthetic_store(date(2024, 2, 16), date(2024, 2, 18))
    labels = source.read_labels()
    labels.loc[labels.index[-1], "y_neg"] = np.nan
    # Storage may return times without a timezone; the join must still match.
    small = MemoryStore()
    small.upsert_features(source.read_features())
    small.upsert_labels(
        labels.assign(delivery_utc=labels["delivery_utc"].dt.tz_localize(None))
    )
    data = tp.load_dataset(small)
    assert data["delivery_utc"].min() == tp.TRAIN_START
    assert data["delivery_utc"].max() < labels["delivery_utc"].max()
    assert data["y_neg"].dtype == "int64"


def test_an_empty_store_fails_clearly() -> None:
    with pytest.raises(ValueError, match="No stored features"):
        tp.load_dataset(MemoryStore())


def test_rolling_windows_are_consecutive_and_end_at_the_newest_label(
    store: MemoryStore,
) -> None:
    parts = tp.rolling_split(tp.load_dataset(store))
    days = {name: sorted(rows["delivery_day"].unique()) for name, rows in parts.items()}
    assert days["holdout"][-1] == "2026-08-31"
    assert len(days["holdout"]) == tp.HOLDOUT_DAYS
    assert len(days["validation"]) == tp.VALIDATION_DAYS
    assert len(days["train"]) == tp.TRAIN_DAYS
    assert days["train"][-1] < days["validation"][0] < days["holdout"][0]


def test_out_of_time_windows_follow_the_plan(store: MemoryStore) -> None:
    parts = tp.out_of_time_split(tp.load_dataset(store))
    assert parts["train"]["delivery_day"].max() == "2024-12-31"
    assert parts["validation"]["delivery_day"].min() == "2025-01-01"
    assert parts["validation"]["delivery_day"].max() == "2025-06-30"
    assert parts["test"]["delivery_day"].min() == "2025-07-01"
    assert parts["test"]["delivery_day"].max() == "2026-06-30"


@pytest.fixture(scope="module")
def retrained(store: MemoryStore, tmp_path_factory: pytest.TempPathFactory):
    registry = LocalRegistry(tmp_path_factory.mktemp("models"))
    result = tp.run("retrain", store=store, registry=registry, params=FAST)
    return result, registry


def test_a_model_that_meets_the_criteria_is_registered(retrained) -> None:
    result, registry = retrained
    assert result.promoted, result.reason
    assert result.model_version == 1
    model, version = registry.load_latest()
    assert version == 1 and model.threshold_ == result.threshold


def test_scores_cover_lightgbm_and_both_baselines(retrained) -> None:
    result, _ = retrained
    for name in ["lgbm", "naive_yesterday", "naive_last_week"]:
        assert f"holdout/{name}" in result.metrics
        assert f"validation/{name}" in result.metrics
    lgbm = result.metrics["holdout/lgbm"]
    assert lgbm["f1"] > result.metrics["holdout/naive_yesterday"]["f1"]


def test_segments_report_season_daypart_and_weekpart(retrained) -> None:
    result, _ = retrained
    segments = result.segments
    assert set(segments["segment"]) == {"season", "daypart", "weekpart"}
    assert set(segments.loc[segments["segment"] == "daypart", "value"]) == {
        "day", "night"}
    for name in ["lgbm", "naive_yesterday", "naive_last_week"]:
        assert {f"{name}_pr_auc", f"{name}_recall"} <= set(segments.columns)
    # Night hours carry almost no planted negatives: counted, not scored.
    night = segments[segments["value"] == "night"].iloc[0]
    if night["positives"] < tp.MIN_SEGMENT_POSITIVES:
        assert np.isnan(night["lgbm_pr_auc"])


def test_calibration_compares_predicted_and_observed_rates(retrained) -> None:
    result, _ = retrained
    assert result.calibration["rows"].sum() == result.metrics["holdout/lgbm"]["rows"]
    assert {"mean_predicted", "observed_rate", "brier"} <= set(result.extra)


def test_summary_states_the_promotion_and_the_version(retrained) -> None:
    result, _ = retrained
    text = "\n".join(tp.summary_lines(result))
    assert "Promotion: yes" in text and "version 1" in text


def test_evaluate_mode_never_registers(store: MemoryStore, tmp_path: Path) -> None:
    registry = LocalRegistry(tmp_path)
    result = tp.run("evaluate", store=store, registry=registry, params=FAST)
    assert not result.promoted and result.model_version is None
    assert "test/lgbm" in result.metrics
    with pytest.raises(NoModelError):
        registry.load_latest()


def test_a_dry_run_registers_nothing(store: MemoryStore) -> None:
    result = tp.run("retrain", store=store, registry=None, params=FAST)
    assert not result.promoted and "dry run" in result.reason


def metrics(pr_auc: float, f1: float, best_baseline: float, positives: int) -> dict:
    return {
        "lgbm": {"pr_auc": pr_auc, "f1": f1, "positives": positives},
        "naive_yesterday": {"f1": best_baseline},
        "naive_last_week": {"f1": best_baseline - 0.1},
    }


@pytest.mark.parametrize(
    ("values", "promoted", "reason"),
    [
        ((0.80, 0.70, 0.50, 100), True, "meet the criteria"),
        ((0.80, 0.70, 0.50, 5), False, "only 5 negative hours"),
        ((0.55, 0.70, 0.50, 100), False, "PR-AUC 0.550 < 0.6"),
        ((0.80, 0.53, 0.50, 100), False, "F1 0.530 < best baseline 0.500"),
    ],
)
def test_promotion_criteria(values: tuple, promoted: bool, reason: str) -> None:
    decision, text = tp.promotion_decision(metrics(*values))
    assert decision is promoted and reason in text


def test_the_mlflow_tracker_logs_metrics_and_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mlflow = pytest.importorskip("mlflow")
    from neghour.tracking import MlflowTracker

    monkeypatch.setattr(config, "DAGSHUB_TOKEN", "")
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    run_id = MlflowTracker().log(
        "test", {"mode": "retrain"}, {"holdout/lgbm/f1": 0.7, "nan": np.nan},
        {"segments": pd.DataFrame({"a": [1]})},
    )
    run = mlflow.get_run(run_id)
    assert run.data.metrics == {"holdout/lgbm/f1": 0.7}
    assert run.data.params == {"mode": "retrain"}


def quiet_recent_store() -> MemoryStore:
    """Two years of data whose last 100 days have no negative hour."""
    store = synthetic_store(date(2024, 3, 1), date(2026, 3, 1), seed=4)
    labels = store.read_labels()
    quiet = labels["delivery_utc"] >= pd.Timestamp("2025-11-22", tz="UTC")
    labels.loc[quiet, "y_neg"] = 0
    store.upsert_labels(labels)
    return store


def test_a_quiet_holdout_grows_back_until_it_has_enough_negative_hours() -> None:
    parts = tp.rolling_split(tp.load_dataset(quiet_recent_store()))
    holdout = parts["holdout"]
    assert holdout["y_neg"].sum() >= tp.MIN_WINDOW_POSITIVES
    assert holdout["delivery_day"].nunique() > tp.HOLDOUT_DAYS
    assert holdout["delivery_day"].nunique() % 7 == tp.HOLDOUT_DAYS % 7
    assert parts["validation"]["y_neg"].sum() >= tp.MIN_WINDOW_POSITIVES
    assert parts["train"]["delivery_day"].max() < parts["validation"][
        "delivery_day"].min()


def test_window_lengths_can_be_set_per_run(store: MemoryStore) -> None:
    parts = tp.rolling_split(tp.load_dataset(store), train_days=180,
                             validation_days=28, holdout_days=28)
    assert parts["train"]["delivery_day"].nunique() == 180
    assert parts["holdout"]["delivery_day"].nunique() == 28


def test_the_summary_reports_the_proposal_targets(retrained) -> None:
    result, _ = retrained
    line = tp.proposal_line(result.metrics, "holdout")
    assert line.startswith("Proposal targets: PR-AUC >= 0.75")
