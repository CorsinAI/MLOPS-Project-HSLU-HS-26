"""Tests for the baseline and LightGBM model wrappers, on synthetic frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neghour.features.validation import FEATURE_COLUMNS
from neghour.models.baselines import NaiveLastWeek, NaiveYesterday
from neghour.models.lgbm_model import (
    MODEL_FEATURES,
    THRESHOLD_GRID,
    NegHourLGBM,
    positive_weight,
)

# Baselines


def test_naive_yesterday_flags_hours_that_were_negative_on_d1() -> None:
    rows = pd.DataFrame(
        {"price_h_d1": [-5.0, 0.0, 12.0, -0.01], "price_h_d7": [9.0, -9.0, -9.0, 9.0]}
    )
    assert NaiveYesterday().predict(rows).tolist() == [1, 0, 0, 1]


def test_naive_last_week_flags_hours_that_were_negative_on_d7() -> None:
    rows = pd.DataFrame(
        {"price_h_d1": [9.0, -9.0, -9.0], "price_h_d7": [-3.0, 0.0, 40.0]}
    )
    assert NaiveLastWeek().predict(rows).tolist() == [1, 0, 0]


@pytest.mark.parametrize("model", [NaiveYesterday(), NaiveLastWeek()])
def test_a_missing_lag_predicts_the_majority_class(model: NaiveYesterday) -> None:
    rows = pd.DataFrame({"price_h_d1": [np.nan, None], "price_h_d7": [np.nan, None]})
    assert model.predict(rows).tolist() == [0, 0]
    assert model.predict_proba(rows).tolist() == [0.0, 0.0]


# LightGBM


def synthetic_rows(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Rows where low residual load at midday makes a negative hour likely."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, len(MODEL_FEATURES)))
    rows = pd.DataFrame(values, columns=MODEL_FEATURES)
    rows["hour_of_day"] = rng.integers(0, 24, n)
    rows["is_holiday"] = rng.random(n) < 0.05
    rows.loc[rng.random(n) < 0.1, "price_h_d7"] = np.nan
    signal = -2.0 * rows["residual_load_proxy"] + (rows["hour_of_day"].between(10, 15))
    y = (signal + rng.normal(scale=0.5, size=n) > 3.0).astype("int64")
    return rows, y


@pytest.fixture(scope="module")
def fitted() -> tuple[NegHourLGBM, pd.DataFrame, pd.Series]:
    X_train, y_train = synthetic_rows(3000, seed=1)
    X_val, y_val = synthetic_rows(1000, seed=2)
    model = NegHourLGBM(params={"n_estimators": 200}).fit(
        X_train, y_train, X_val, y_val
    )
    return model, X_val, y_val


def test_model_features_are_stored_features_without_bookkeeping() -> None:
    assert set(MODEL_FEATURES) <= set(FEATURE_COLUMNS)
    for column in MODEL_FEATURES:
        assert not column.startswith(("available_at", "missing_"))
        bookkeeping = {"delivery_utc", "delivery_day", "cutoff_utc", "fetched_at"}
        assert column not in bookkeeping


def test_the_synthetic_label_is_rare() -> None:
    _, y = synthetic_rows(3000, seed=1)
    assert 0.02 < y.mean() < 0.15


def test_scale_pos_weight_is_negatives_over_positives(
    fitted: tuple[NegHourLGBM, pd.DataFrame, pd.Series],
) -> None:
    model, _, _ = fitted
    _, y_train = synthetic_rows(3000, seed=1)
    expected = (y_train == 0).sum() / (y_train == 1).sum()
    assert model.scale_pos_weight_ == pytest.approx(expected)
    assert model.model_.get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_training_without_a_negative_hour_fails_clearly() -> None:
    with pytest.raises(ValueError, match="no negative-price hour"):
        positive_weight(pd.Series([0, 0, 0]))


def test_probabilities_are_in_the_unit_interval(
    fitted: tuple[NegHourLGBM, pd.DataFrame, pd.Series],
) -> None:
    model, X_val, _ = fitted
    scores = model.predict_proba(X_val)
    assert scores.shape == (len(X_val),)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_the_model_learns_the_planted_signal(
    fitted: tuple[NegHourLGBM, pd.DataFrame, pd.Series],
) -> None:
    from sklearn.metrics import average_precision_score

    model, X_val, y_val = fitted
    assert average_precision_score(y_val, model.predict_proba(X_val)) > 0.6


def test_the_threshold_comes_from_the_validation_grid(
    fitted: tuple[NegHourLGBM, pd.DataFrame, pd.Series],
) -> None:
    model, X_val, y_val = fitted
    assert 0.10 <= model.threshold_ <= 0.90
    assert model.threshold_ in THRESHOLD_GRID
    assert model.threshold_ == model.threshold_from_validation(X_val, y_val)


def test_predict_uses_the_tuned_threshold_unless_given_one(
    fitted: tuple[NegHourLGBM, pd.DataFrame, pd.Series],
) -> None:
    model, X_val, _ = fitted
    scores = model.predict_proba(X_val)
    assert (model.predict(X_val) == (scores >= model.threshold_)).all()
    assert (model.predict(X_val, threshold=0.99) == (scores >= 0.99)).all()


def test_extra_columns_are_ignored(
    fitted: tuple[NegHourLGBM, pd.DataFrame, pd.Series],
) -> None:
    model, X_val, _ = fitted
    wider = X_val.assign(missing_price=True, delivery_day="2026-11-25")
    np.testing.assert_allclose(model.predict_proba(wider), model.predict_proba(X_val))


def test_feature_importance_lists_every_feature(
    fitted: tuple[NegHourLGBM, pd.DataFrame, pd.Series],
) -> None:
    model, _, _ = fitted
    importance = model.feature_importance()
    assert sorted(importance["feature"]) == sorted(MODEL_FEATURES)
    assert importance.loc[0, "feature"] == "residual_load_proxy"


def test_save_and_load_give_the_same_predictions(
    fitted: tuple[NegHourLGBM, pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    model, X_val, _ = fitted
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = NegHourLGBM.load(path)
    assert loaded.threshold_ == model.threshold_
    np.testing.assert_allclose(loaded.predict_proba(X_val), model.predict_proba(X_val))


def test_an_unfitted_model_refuses_to_predict() -> None:
    X, _ = synthetic_rows(5, seed=3)
    with pytest.raises(RuntimeError, match="not fitted"):
        NegHourLGBM().predict_proba(X)
