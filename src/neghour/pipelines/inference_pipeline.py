"""
Inference pipeline: predict every delivery hour of day D and store the result.

Steps:
  1. Refuse to run before the cut-off (10:00 Berlin on D-1): features built
     earlier miss data that counts as available by the cut-off.
  2. Rebuild the feature rows for D with the feature pipeline. Before gate
     closure (12:00 Berlin on D-1) this replaces rows built in the early
     morning; after it, the stored rows are frozen and stay as they are
     (decision #45). If the sources are down, the stored rows are used.
  3. Load the newest registered model (only promoted models are registered).
  4. Predict P(price < 0) and yes or no for each of the 23, 24 or 25 hours.
  5. Store the predictions, keyed by delivery_utc, with `predicted_at` and
     `on_time` (predicted at or before gate closure). A late run never replaces
     an on-time prediction of the same day.

Outcomes are not copied here. Scores join predictions to labels on
delivery_utc (README, "Monitoring plan").

The time-critical part is the start: the run must start after 10:00 and finish
before 12:00 Berlin. GitHub cron cannot promise that, so this module only
offers an entry point; the trigger is chosen separately (decision #47).

Triggered by: .github/workflows/inference.yml (manual or external trigger)
Invoked via:  python -m neghour.pipelines.inference_pipeline [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from neghour.config import BERLIN_TZ
from neghour.features.validation import validate_day_lengths, validate_no_leakage
from neghour.models.registry import ModelRegistry, get_registry
from neghour.pipelines import feature_pipeline
from neghour.sources.http import SourceUnavailableError
from neghour.storage import get_store
from neghour.storage.base import FeatureStore
from neghour.timeutils import berlin_today, cutoff_utc, gate_closure_utc

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS: list[str] = [
    "delivery_utc",
    "delivery_day",
    "p_neg",
    "y_pred",
    "threshold",
    "model_version",
    "predicted_at",
    "gate_closure_utc",
    "on_time",
    "missing_inputs",
]
MISSING_FLAGS = ["missing_price", "missing_weather", "missing_actuals"]
TIME_COLUMNS = ["delivery_utc", "cutoff_utc", "available_at", "available_at_price",
                "available_at_weather", "available_at_actuals",
                "available_at_calendar", "fetched_at"]


class TooEarlyError(RuntimeError):
    """Raised when inference for D is started before the cut-off on D-1."""


@dataclass(frozen=True)
class InferenceSummary:
    delivery_day: date
    rows: int
    negative_hours: int
    model_version: int
    predicted_at: datetime
    on_time: bool
    written: bool
    rows_with_missing_inputs: int
    note: str = ""


def _with_utc_times(frame: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in TIME_COLUMNS if c in frame]
    return frame.assign(**{c: pd.to_datetime(frame[c], utc=True) for c in present})


def feature_rows(store: FeatureStore, day: date) -> pd.DataFrame:
    """Return the stored feature rows of one delivery day, checked again."""
    stored = store.read_features()
    if stored.empty:
        raise ValueError("No stored feature rows")
    rows = stored[stored["delivery_day"].astype(str) == day.isoformat()]
    if rows.empty:
        raise ValueError(f"No stored feature rows for {day}")
    rows = _with_utc_times(rows).sort_values("delivery_utc").reset_index(drop=True)
    validate_day_lengths(rows)
    validate_no_leakage(rows)
    return rows


def has_on_time_prediction(stored: pd.DataFrame, day: date) -> bool:
    if stored.empty:
        return False
    rows = stored[stored["delivery_day"].astype(str) == day.isoformat()]
    return bool(rows["on_time"].astype(bool).any())


def run(
    delivery_date: date | None = None,
    store: FeatureStore | None = None,
    registry: ModelRegistry | None = None,
    now: datetime | None = None,
    refresh: bool = True,
    write: bool = True,
) -> tuple[InferenceSummary, pd.DataFrame]:
    """Predict *delivery_date* (default: Berlin tomorrow) and store the result."""
    now = now if now is not None else datetime.now(UTC)
    day = delivery_date if delivery_date is not None else (
        berlin_today(now) + timedelta(days=1)
    )
    if pd.Timestamp(now) < cutoff_utc(day):
        raise TooEarlyError(
            f"Inference for {day} may start at {cutoff_utc(day)} UTC at the earliest"
        )
    target = store if store is not None else get_store()
    models = registry if registry is not None else get_registry()

    if refresh:
        try:
            feature_pipeline.run(day, day, target, tolerate_missing=True, now=now)
        except SourceUnavailableError as error:
            logger.warning("Feature refresh failed, using stored rows: %s", error)

    rows = feature_rows(target, day)
    model, version = models.load_latest()
    proba = model.predict_proba(rows)
    gate = gate_closure_utc(day)
    predicted_at = pd.Timestamp(now).tz_convert("UTC")
    on_time = predicted_at <= gate
    predictions = pd.DataFrame(
        {
            "delivery_utc": rows["delivery_utc"],
            "delivery_day": day.isoformat(),
            "p_neg": proba.astype("float64"),
            "y_pred": (proba >= model.threshold_).astype("int64"),
            "threshold": float(model.threshold_),
            "model_version": int(version),
            "predicted_at": predicted_at,
            "gate_closure_utc": gate,
            "on_time": on_time,
            "missing_inputs": rows[MISSING_FLAGS].astype(bool).any(axis=1),
        }
    )[PREDICTION_COLUMNS]

    written, note = False, ""
    if not write:
        note = "dry run: nothing written"
    elif not on_time and has_on_time_prediction(target.read_predictions(), day):
        note = "late run: the stored on-time prediction is kept"
    else:
        target.upsert_predictions(predictions)
        written = True
    if not on_time:
        logger.warning("Prediction for %s made after gate closure %s", day, gate)

    summary = InferenceSummary(
        delivery_day=day,
        rows=len(predictions),
        negative_hours=int(predictions["y_pred"].sum()),
        model_version=int(version),
        predicted_at=predicted_at.to_pydatetime(),
        on_time=bool(on_time),
        written=written,
        rows_with_missing_inputs=int(predictions["missing_inputs"].sum()),
        note=note,
    )
    return summary, predictions


def write_step_summary(summary: InferenceSummary, predictions: pd.DataFrame) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    berlin = summary.predicted_at.astimezone(BERLIN_TZ)
    hours = predictions.assign(
        hour=pd.to_datetime(predictions["delivery_utc"], utc=True)
        .dt.tz_convert(BERLIN_TZ)
        .dt.strftime("%H:%M")
    )
    lines = [
        "## Inference",
        "",
        f"- Delivery day: {summary.delivery_day}",
        f"- Predicted at: {berlin:%Y-%m-%d %H:%M} Berlin, "
        f"{'on time' if summary.on_time else 'AFTER gate closure'}",
        f"- Model version: {summary.model_version}",
        f"- Negative hours predicted: {summary.negative_hours} of {summary.rows}",
        f"- Rows with missing inputs: {summary.rows_with_missing_inputs}",
        f"- Written: {'yes' if summary.written else 'no'} {summary.note}",
        "",
        "```",
        hours[["hour", "p_neg", "y_pred"]].round(3).to_string(index=False),
        "```",
    ]
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument("--no-refresh", action="store_true",
                        help="use the stored feature rows as they are")
    parser.add_argument("--dry-run", action="store_true",
                        help="refresh nothing and write nothing, only print")
    args = parser.parse_args(argv)

    summary, predictions = run(
        args.date, refresh=not (args.no_refresh or args.dry_run),
        write=not args.dry_run,
    )
    logger.info("Inference finished: %s", summary)
    print(predictions[["delivery_utc", "p_neg", "y_pred"]].round(3).to_string(
        index=False))
    write_step_summary(summary, predictions)


if __name__ == "__main__":
    main()
