"""
Feature pipeline: fetch sources, build rows, validate, upsert.

The daily run rebuilds Berlin today - 7 days through Berlin tomorrow, so missed
days repair themselves and labels of past days stay fresh. Rows are selected by
timestamp, so a late run writes the same features as an on-time run.

Late or missing data: a source that still fails after retries gives NaN plus a
`missing_*` flag, a warning and a step-summary line. Values are never carried
forward. During an outage only the newest delivery day is written, so good
stored rows are not overwritten, and the next run repairs that day. The run
fails if storage or validation fails, or if SMARD and Open-Meteo are both down.
A backfill passes tolerate_missing=False, so it fails instead of storing gaps.

Invoked via:  python -m neghour.pipelines.feature_pipeline
Triggered by: .github/workflows/daily_pipeline.yml
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TypeVar

from neghour.features.build import NEGATIVE_WINDOW_DAYS, build_features, build_labels
from neghour.features.validation import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    validate_day_lengths,
    validate_no_leakage,
    validate_schema,
    validate_unique_hours,
)
from neghour.sources import open_meteo, smard
from neghour.sources.http import SourceUnavailableError
from neghour.storage import get_store
from neghour.storage.base import FeatureStore
from neghour.timeutils import berlin_today, delivery_hours

logger = logging.getLogger(__name__)

DAILY_LOOKBACK_DAYS = 7

T = TypeVar("T")


@dataclass(frozen=True)
class RunSummary:
    """What one pipeline run did."""

    first_day: date
    last_day: date
    feature_rows: int
    label_rows: int
    missing_price_rows: int
    missing_weather_rows: int
    missing_actuals_rows: int
    duration_s: float
    failed_sources: tuple[str, ...] = ()


def daily_window(now: datetime | None = None) -> tuple[date, date]:
    """Return the delivery days a daily run rebuilds."""
    today = berlin_today(now)
    return today - timedelta(days=DAILY_LOOKBACK_DAYS), today + timedelta(days=1)


def run(
    first_day: date,
    last_day: date,
    store: FeatureStore | None = None,
    tolerate_missing: bool = False,
) -> RunSummary:
    """Build and store features and labels for the given delivery days."""
    started = time.monotonic()
    target = store if store is not None else get_store()
    failed: list[str] = []

    def fetch(name: str, call: Callable[[], T], fallback: Callable[[], T]) -> T:
        try:
            return call()
        except SourceUnavailableError:
            if not tolerate_missing:
                raise
            logger.warning("Source unavailable, continuing without it: %s", name)
            failed.append(name)
            return fallback()

    fetch_start = first_day - timedelta(days=NEGATIVE_WINDOW_DAYS)
    prices = fetch(
        "smard_prices",
        lambda: smard.fetch_prices(fetch_start, last_day),
        smard.empty_prices,
    )
    quarters = fetch(
        "smard_quarters",
        lambda: smard.fetch_negative_quarters(first_day, last_day),
        smard.empty_negative_quarters,
    )
    actuals = fetch(
        "smard_actuals",
        lambda: smard.fetch_actuals(fetch_start, last_day),
        smard.empty_actuals,
    )
    weather = fetch(
        "open_meteo",
        lambda: open_meteo.fetch_weather(first_day, last_day),
        open_meteo.empty_weather,
    )
    if {"smard_prices", "open_meteo"} <= set(failed):
        raise SourceUnavailableError("SMARD and Open-Meteo are both unavailable")

    features = build_features(prices, weather, actuals, first_day, last_day)
    in_window = prices["delivery_utc"].isin(delivery_hours(first_day, last_day))
    labels = build_labels(prices[in_window], quarters)

    validate_schema(features, FEATURE_COLUMNS)
    validate_schema(labels, LABEL_COLUMNS)
    validate_unique_hours(features)
    validate_unique_hours(labels)
    validate_day_lengths(features)
    validate_no_leakage(features)

    if failed:
        features = features[features["delivery_day"] == last_day.isoformat()]

    target.upsert_features(features)
    if not labels.empty:
        target.upsert_labels(labels)

    return RunSummary(
        first_day=first_day,
        last_day=last_day,
        feature_rows=len(features),
        label_rows=len(labels),
        missing_price_rows=int(features["missing_price"].sum()),
        missing_weather_rows=int(features["missing_weather"].sum()),
        missing_actuals_rows=int(features["missing_actuals"].sum()),
        duration_s=round(time.monotonic() - started, 1),
        failed_sources=tuple(failed),
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
        f"| Rows with missing weather inputs | {summary.missing_weather_rows} |",
        f"| Rows with missing actuals | {summary.missing_actuals_rows} |",
        f"| Failed sources | {', '.join(summary.failed_sources) or 'none'} |",
        f"| Duration (s) | {summary.duration_s} |",
    ]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    first_day, last_day = daily_window()
    summary = run(first_day, last_day, tolerate_missing=True)
    logger.info("Run finished: %s", summary)
    write_step_summary(summary)


if __name__ == "__main__":
    main()
