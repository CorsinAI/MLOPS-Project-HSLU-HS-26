"""
Schema and leakage validation for assembled feature DataFrames.

All checks run before data is written to the Hopsworks feature group.
A leakage violation raises LeakageError and aborts the pipeline run.
"""

from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS: list[str] = [
    "delivery_date",
    "delivery_hour",
    "available_at",
    # weather
    "irradiance",
    "cloud_cover",
    "wind_100m",
    "temperature",
    "residual_load_proxy",
    # market
    "price_h_d1",
    "price_h_d7",
    "neg_hours_last_7d",
    "load_d2",
    # lags
    "solar_d2",
    "wind_d2",
    # calendar
    "hour_of_day",
    "weekday",
    "is_holiday",
    "month",
    # label (None / NaN during inference; populated after D-1 12:45)
    "y_neg",
]


class LeakageError(ValueError):
    """Raised when a feature row contains data published after the cut-off."""


# TODO: implement validate_schema(df: pd.DataFrame) -> None
#       raise ValueError listing missing / extra columns
# TODO: implement validate_no_leakage(df: pd.DataFrame, cutoff_hour: int = 10) -> None
#       raise LeakageError if any available_at >= D-1 cutoff_hour:00 Berlin
# TODO: implement validate_label_not_in_features(df: pd.DataFrame) -> None
#       ensure y_neg is absent or all-NaN in the feature columns passed to the model
