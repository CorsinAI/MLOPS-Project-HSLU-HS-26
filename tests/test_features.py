"""Tests for label and feature assembly and leakage validation."""

from datetime import date, timedelta

import pandas as pd
import pytest

from neghour.availability import price_available_at, weather_available_at
from neghour.features.build import build_features as build_all
from neghour.features.build import build_labels
from neghour.features.validation import (
    FEATURE_COLUMNS,
    LeakageError,
    validate_no_leakage,
    validate_schema,
)
from neghour.timeutils import delivery_hours

DAY = date(2026, 1, 15)


def weather_frame(first_day: date, last_day: date) -> pd.DataFrame:
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


def build_features(prices: pd.DataFrame, first: date, last: date) -> pd.DataFrame:
    return build_all(prices, weather_frame(first, last), first, last)


def price_frame(first_day: date, last_day: date, value: float = 50.0) -> pd.DataFrame:
    hours = delivery_hours(first_day, last_day)
    return pd.DataFrame(
        {
            "delivery_utc": hours,
            "price_eur_mwh": value,
            "available_at": price_available_at(hours),
        }
    )


def set_price(prices: pd.DataFrame, moment: pd.Timestamp, value: float) -> None:
    prices.loc[prices["delivery_utc"] == moment, "price_eur_mwh"] = value


def test_label_is_strictly_below_zero() -> None:
    prices = price_frame(DAY, DAY).head(3)
    prices["price_eur_mwh"] = [-0.01, 0.0, 0.01]
    assert build_labels(prices)["y_neg"].tolist() == [1, 0, 0]


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
    features = build_all(prices, weather, DAY, DAY)
    assert features["wind_100m"].isna().all()
    assert features["missing_weather"].all()
    validate_no_leakage(features)


def test_missing_weather_keeps_the_row_with_a_flag() -> None:
    prices = price_frame(DAY - timedelta(days=7), DAY)
    empty = weather_frame(DAY, DAY).head(0)
    features = build_all(prices, empty, DAY, DAY)
    assert len(features) == 24
    assert features["missing_weather"].all()
    assert features["price_h_d1"].notna().all()
