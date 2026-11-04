"""
Schema and leakage validation. Runs before anything is written to storage.

A leakage violation raises LeakageError and aborts the run.
"""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "delivery_utc",
    "delivery_day",
    "cutoff_utc",
    "available_at",
    # price features
    "price_h_d1",
    "price_h_d7",
    "neg_hours_last_7d",
    "available_at_price",
    "missing_price",
]

LABEL_COLUMNS: list[str] = [
    "delivery_utc",
    "price_eur_mwh",
    "y_neg",
    "available_at",
]

AVAILABILITY_COLUMNS: list[str] = [
    "available_at",
    "available_at_price",
]


class LeakageError(ValueError):
    """Raised when a feature row contains data published after the cut-off."""


def validate_schema(frame: pd.DataFrame, expected: list[str]) -> None:
    """Raise ValueError if the frame's columns differ from the expected ones."""
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {missing}. Extra: {extra}.")


def validate_no_leakage(features: pd.DataFrame) -> None:
    """Raise LeakageError if any availability timestamp is after the cut-off."""
    for column in AVAILABILITY_COLUMNS:
        late = features[features[column] > features["cutoff_utc"]]
        if not late.empty:
            first = late["delivery_utc"].min()
            raise LeakageError(
                f"{len(late)} rows have {column} after the cut-off, first at {first}."
            )
