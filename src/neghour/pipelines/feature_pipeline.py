"""
Feature pipeline: fetch sources, build rows, validate, upsert.

The daily run rebuilds Berlin today - 7 days through Berlin tomorrow, so missed
days repair themselves and labels of past days stay fresh. Rows are selected by
timestamp, so a late run writes the same features as an on-time run.

Invoked via:  python -m neghour.pipelines.feature_pipeline
Triggered by: .github/workflows/daily_pipeline.yml
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from neghour.features.build import NEGATIVE_WINDOW_DAYS, build_features, build_labels
from neghour.features.validation import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    validate_no_leakage,
    validate_schema,
)
from neghour.sources import smard
from neghour.storage import get_store
from neghour.storage.base import FeatureStore
from neghour.timeutils import berlin_today, delivery_hours

logger = logging.getLogger(__name__)

DAILY_LOOKBACK_DAYS = 7


@dataclass(frozen=True)
class RunSummary:
    """What one pipeline run did."""

    first_day: date
    last_day: date
    feature_rows: int
    label_rows: int
    missing_price_rows: int
    duration_s: float


def daily_window(now: datetime | None = None) -> tuple[date, date]:
    """Return the delivery days a daily run rebuilds."""
    today = berlin_today(now)
    return today - timedelta(days=DAILY_LOOKBACK_DAYS), today + timedelta(days=1)


def run(
    first_day: date, last_day: date, store: FeatureStore | None = None
) -> RunSummary:
    """Build and store features and labels for the given delivery days."""
    started = time.monotonic()
    target = store if store is not None else get_store()

    fetch_start = first_day - timedelta(days=NEGATIVE_WINDOW_DAYS)
    prices = smard.fetch_prices(fetch_start, last_day)

    features = build_features(prices, first_day, last_day)
    in_window = prices["delivery_utc"].isin(delivery_hours(first_day, last_day))
    labels = build_labels(prices[in_window])

    validate_schema(features, FEATURE_COLUMNS)
    validate_schema(labels, LABEL_COLUMNS)
    validate_no_leakage(features)

    target.upsert_features(features)
    target.upsert_labels(labels)

    return RunSummary(
        first_day=first_day,
        last_day=last_day,
        feature_rows=len(features),
        label_rows=len(labels),
        missing_price_rows=int(features["missing_price"].sum()),
        duration_s=round(time.monotonic() - started, 1),
    )


def write_step_summary(summary: RunSummary) -> None:
    """Add a run overview to the GitHub Actions job page, if running there."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = [
        "## Feature pipeline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Delivery days | {summary.first_day} to {summary.last_day} |",
        f"| Feature rows | {summary.feature_rows} |",
        f"| Label rows | {summary.label_rows} |",
        f"| Rows with missing price inputs | {summary.missing_price_rows} |",
        f"| Duration (s) | {summary.duration_s} |",
    ]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    first_day, last_day = daily_window()
    summary = run(first_day, last_day)
    logger.info("Run finished: %s", summary)
    write_step_summary(summary)


if __name__ == "__main__":
    main()
