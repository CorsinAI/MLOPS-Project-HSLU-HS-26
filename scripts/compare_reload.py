"""Measure how far stored feature rows differ from a fresh reload (review A1).

Reads the stored feature rows for a range of delivery days, rebuilds the same
days from the sources with the backfill code into memory, and reports how often
the values differ, per model feature and per row. Nothing is written.

By default it reads features version 1, which still holds rows rewritten for
about a week after their cut-off. A difference therefore means a source changed
a value after that last rewrite.

Usage:
  python scripts/compare_reload.py --first-day 2026-10-01 --last-day 2026-11-18
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from neghour import config
from neghour.models.lgbm_model import MODEL_FEATURES
from neghour.pipelines import feature_pipeline
from neghour.storage import get_store
from neghour.storage.memory import MemoryStore

FLAGS = ["missing_price", "missing_weather", "missing_actuals"]
TOLERANCE = 1e-6


def differences(
    stored: pd.DataFrame, reloaded: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Return one boolean column per feature: True where the two rows differ.

    Rows are matched on delivery_utc; only hours present in both are compared.
    Two NaN values count as equal, a NaN against a number as different.
    """
    left = stored.assign(delivery_utc=pd.to_datetime(stored["delivery_utc"], utc=True))
    right = reloaded.assign(
        delivery_utc=pd.to_datetime(reloaded["delivery_utc"], utc=True)
    )
    both = left.merge(right, on="delivery_utc", suffixes=("_s", "_r"))
    result = pd.DataFrame({"delivery_utc": both["delivery_utc"]})
    for column in columns:
        a = both[f"{column}_s"].astype("float64").to_numpy()
        b = both[f"{column}_r"].astype("float64").to_numpy()
        same = np.isclose(a, b, rtol=0, atol=TOLERANCE) | (np.isnan(a) & np.isnan(b))
        result[column] = ~same
    return result


def report(diff: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return report lines: share of hours that differ, per column and overall."""
    hours = len(diff)
    lines = [f"Hours compared: {hours}"]
    if hours == 0:
        return lines
    any_change = diff[columns].any(axis=1)
    lines.append(
        f"Hours with any difference: {int(any_change.sum())} "
        f"({any_change.mean():.1%})"
    )
    for column in columns:
        changed = int(diff[column].sum())
        if changed:
            lines.append(f"  {column}: {changed} ({changed / hours:.1%})")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-day", type=date.fromisoformat, required=True)
    parser.add_argument("--last-day", type=date.fromisoformat, required=True)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    config.FEATURE_GROUP_VERSION = args.version
    stored = get_store().read_features()
    stored = stored[
        stored["delivery_day"].between(
            args.first_day.isoformat(), args.last_day.isoformat()
        )
    ]
    memory = MemoryStore()
    feature_pipeline.run(
        args.first_day, args.last_day, memory, now=datetime.now(UTC)
    )
    columns = MODEL_FEATURES + FLAGS
    diff = differences(stored, memory.read_features(), columns)
    print(f"Stored version {args.version}, {args.first_day} to {args.last_day}")
    print("\n".join(report(diff, columns)))


if __name__ == "__main__":
    main()
