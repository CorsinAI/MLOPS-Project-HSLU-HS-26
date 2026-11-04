"""Tests for label and feature assembly and leakage validation."""

from datetime import date, timedelta

import pandas as pd
import pytest

from neghour.features.build import build_features as build_all
from neghour.features.build import build_labels
from neghour.features.validation import (
    AVAILABILITY_COLUMNS,
    FEATURE_COLUMNS,
    LeakageError,
    validate_no_leakage,
    validate_schema,
)
from tests.frames import actuals_frame, price_frame, weather_frame

DAY = date(2026, 1, 15)


def build_features(prices: pd.DataFrame, first: date, last: date) -> pd.DataFrame:
    actuals = actuals_frame(first - timedelta(days=7), last)
    return build_all(prices, weather_frame(first, last), actuals, first, last)


def no_quarters() -> pd.Series:
    return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))


def set_price(prices: pd.DataFrame, moment: pd.Timestamp, value: float) -> None:
    prices.loc[prices["delivery_utc"] == moment, "price_eur_mwh"] = value


def test_label_is_strictly_below_zero() -> None:
    prices = price_frame(DAY, DAY).head(3)
    prices["price_eur_mwh"] = [-0.01, 0.0, 0.01]
    assert build_labels(prices, no_quarters())["y_neg"].tolist() == [1, 0, 0]


def test_labels_keep_quarter_counts_where_they_exist() -> None:
    prices = price_frame(DAY, DAY).head(2)
    first_hour = pd.DatetimeIndex(prices["delivery_utc"].iloc[:1])
    labels = build_labels(prices, pd.Series([3.0], index=first_hour))
    assert labels["n_neg_quarters"].iloc[0] == 3.0
    assert pd.isna(labels["n_neg_quarters"].iloc[1])


def test_price_lags_use_same_hour_on_d1_and_d7() -> None:
    prices = price_frame(DAY - timedelta(days=7), DAY)
    noon = pd.Timestamp("2026-01-15 11:00", tz="UTC")
    set_price(prices, noon - pd.Timedelta(days=1), 11.0)
    set_price(prices, noon - pd.Timedelta(days=7), 77.0)
    features = build_features(prices, DAY, DAY).set_index("delivery_utc")
    assert features.loc[noon, "price_h_d1"] == 11.0
    assert features.loc[noon, "price_h_d7"] == 77.0


def test_negative_hour_count_covers_the_seven_days_before_delivery() -> None:
    prices = price_frame(DAY - timedelta(days=8), DAY)
    day_minus_one = pd.Timestamp("2026-01-14 11:00", tz="UTC")
    day_minus_seven = pd.Timestamp("2026-01-08 11:00", tz="UTC")
    day_minus_eight = pd.Timestamp("2026-01-07 11:00", tz="UTC")
    delivery_day = pd.Timestamp("2026-01-15 11:00", tz="UTC")
    for moment in (day_minus_one, day_minus_seven, day_minus_eight, delivery_day):
        set_price(prices, moment, -5.0)
    features = build_features(prices, DAY, DAY)
    assert (features["neg_hours_last_7d"] == 2.0).all()


def test_delivery_day_prices_never_reach_features() -> None:
    base = price_frame(DAY - timedelta(days=7), DAY)
    changed = base.copy()
    cutoff = pd.Timestamp("2026-01-14 09:00", tz="UTC")
    changed.loc[changed["available_at"] > cutoff, "price_eur_mwh"] = -999.0
    pd.testing.assert_frame_equal(
        build_features(base, DAY, DAY), build_features(changed, DAY, DAY)
    )


def test_missing_input_gives_nan_and_a_flag() -> None:
    prices = price_frame(DAY - timedelta(days=1), DAY)
    features = build_features(prices, DAY, DAY)
    assert features["price_h_d7"].isna().all()
    assert features["missing_price"].all()


def test_features_match_the_schema_and_pass_the_cutoff_check() -> None:
    features = build_features(price_frame(DAY - timedelta(days=7), DAY), DAY, DAY)
    validate_schema(features, FEATURE_COLUMNS)
    validate_no_leakage(features)


def test_validate_schema_names_missing_columns() -> None:
    with pytest.raises(ValueError, match="price_h_d1"):
        validate_schema(pd.DataFrame({"delivery_utc": []}), ["price_h_d1"])


def test_validate_no_leakage_rejects_a_late_timestamp() -> None:
    features = build_features(price_frame(DAY - timedelta(days=7), DAY), DAY, DAY)
    features.loc[0, "available_at"] = features.loc[0, "cutoff_utc"] + pd.Timedelta("1s")
    with pytest.raises(LeakageError):
        validate_no_leakage(features)


def test_validate_no_leakage_accepts_a_timestamp_exactly_at_the_cutoff() -> None:
    features = build_features(price_frame(DAY - timedelta(days=7), DAY), DAY, DAY)
    features.loc[0, "available_at"] = features.loc[0, "cutoff_utc"]
    validate_no_leakage(features)


def test_weather_published_after_the_cutoff_is_dropped() -> None:
    prices = price_frame(DAY - timedelta(days=7), DAY)
    weather = weather_frame(DAY, DAY)
    weather["available_at_weather"] = pd.Timestamp("2026-01-14 09:00:01", tz="UTC")
    actuals = actuals_frame(DAY - timedelta(days=7), DAY)
    features = build_all(prices, weather, actuals, DAY, DAY)
    assert features["wind_100m"].isna().all()
    assert features["missing_weather"].all()
    validate_no_leakage(features)


def test_missing_weather_keeps_the_row_with_a_flag() -> None:
    prices = price_frame(DAY - timedelta(days=7), DAY)
    empty = weather_frame(DAY, DAY).head(0)
    actuals = actuals_frame(DAY - timedelta(days=7), DAY)
    features = build_all(prices, empty, actuals, DAY, DAY)
    assert len(features) == 24
    assert features["missing_weather"].all()
    assert features["price_h_d1"].notna().all()


def test_actuals_use_same_hour_on_d2_and_d7() -> None:
    prices = price_frame(DAY - timedelta(days=7), DAY)
    actuals = actuals_frame(DAY - timedelta(days=7), DAY)
    noon = pd.Timestamp("2026-01-15 11:00", tz="UTC")
    day_2 = actuals["delivery_utc"] == noon - pd.Timedelta(days=2)
    day_7 = actuals["delivery_utc"] == noon - pd.Timedelta(days=7)
    actuals.loc[day_2, ["load_mwh", "solar_mwh", "wind_mwh"]] = [1.0, 2.0, 3.0]
    actuals.loc[day_7, "load_mwh"] = 7.0
    weather = weather_frame(DAY, DAY)
    features = build_all(prices, weather, actuals, DAY, DAY).set_index("delivery_utc")
    row = features.loc[noon]
    assert (row["load_d2"], row["solar_d2"], row["wind_d2"]) == (1.0, 2.0, 3.0)
    assert row["load_h_d7"] == 7.0


def test_actuals_published_after_the_cutoff_are_dropped() -> None:
    prices = price_frame(DAY - timedelta(days=7), DAY)
    actuals = actuals_frame(DAY - timedelta(days=7), DAY)
    actuals["available_at_actuals"] = pd.Timestamp("2026-01-14 09:00:01", tz="UTC")
    features = build_all(prices, weather_frame(DAY, DAY), actuals, DAY, DAY)
    assert features["load_d2"].isna().all()
    assert features["missing_actuals"].all()
    assert features["residual_load_proxy"].isna().all()
    validate_no_leakage(features)


def test_residual_load_proxy_is_load_minus_expected_renewables() -> None:
    prices = price_frame(DAY - timedelta(days=7), DAY)
    features = build_features(prices, DAY, DAY)
    expected = 60_000 - 73_000 * 0.5**3 - 100_000 * 0.5
    assert features["residual_load_proxy"].iloc[0] == pytest.approx(expected)


def test_every_availability_column_is_filled_for_a_complete_row() -> None:
    prices = price_frame(DAY - timedelta(days=7), DAY)
    features = build_features(prices, DAY, DAY)
    assert features[AVAILABILITY_COLUMNS].notna().all().all()
