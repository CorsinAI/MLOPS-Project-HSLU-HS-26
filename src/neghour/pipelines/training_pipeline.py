"""
Training pipeline: fit LightGBM, compare it with both baselines, promote it.

Steps:
  1. Read features and labels, join them on delivery_utc, and keep labelled
     rows from 2024-02-17 06:00 UTC on (decision #38).
  2. Split by delivery day (see below). Validation data is used for early
     stopping and the F1 threshold, never for the scores that decide anything.
  3. Fit LightGBM with scale_pos_weight (README, "Class imbalance").
  4. Score LightGBM and both baselines: PR-AUC, F1, precision and recall. Also
     per season, day and night, weekday and weekend (review E8), and a
     calibration table, because scale_pos_weight shifts the probabilities.
  5. Log the run to MLflow (neghour.tracking).
  6. Retrain mode only: register the model if it meets the criteria.

Two modes:
  evaluate  Out-of-time, once: train to the end of 2024, validate on H1 2025,
            test on Jul 2025 to Jun 2026. It never registers, so the test year
            is not used for model selection.
  retrain   Rolling window (decision #37): the newest HOLDOUT_DAYS decide
            promotion, the VALIDATION_DAYS before them tune the model, and the
            TRAIN_DAYS before those train it. Promotion needs PR-AUC >= 0.75
            and F1 >= best baseline F1 + 0.10 on the holdout, and at least
            MIN_HOLDOUT_POSITIVES negative hours there; otherwise the current
            model stays.

Triggered by: .github/workflows/weekly_training.yml (manual for now)
Invoked via:  python -m neghour.pipelines.training_pipeline --mode retrain
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from neghour import config
from neghour.models.baselines import BASELINES
from neghour.models.lgbm_model import NegHourLGBM
from neghour.models.registry import ModelRegistry, get_registry
from neghour.storage import get_store
from neghour.storage.base import FeatureStore
from neghour.tracking import MlflowTracker, NullTracker, Tracker

logger = logging.getLogger(__name__)

TRAIN_START = pd.Timestamp("2024-02-17 06:00", tz="UTC")

# Out-of-time split, by Berlin delivery day; each end is exclusive.
OOT_VALIDATION_START = "2025-01-01"
OOT_TEST_START = "2025-07-01"
OOT_TEST_END = "2026-07-01"

# Rolling window (decision #46). Whole weeks keep the weekday mix even.
TRAIN_DAYS = 364
VALIDATION_DAYS = 56
HOLDOUT_DAYS = 56
MIN_HOLDOUT_POSITIVES = 20

# A segment with fewer negative hours is reported, but not scored.
MIN_SEGMENT_POSITIVES = 10
DAY_HOURS = range(8, 18)  # 08:00 to 17:59 Berlin
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
           5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn",
           10: "autumn", 11: "autumn"}
CALIBRATION_BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


@dataclass
class TrainingResult:
    mode: str
    windows: dict[str, tuple[str, str]]
    metrics: dict[str, dict[str, float]]
    segments: pd.DataFrame
    calibration: pd.DataFrame
    importance: pd.DataFrame
    threshold: float
    promoted: bool = False
    reason: str = ""
    model_version: int | None = None
    run_id: str | None = None
    extra: dict[str, float] = field(default_factory=dict)


# Data


def load_dataset(store: FeatureStore) -> pd.DataFrame:
    """Return labelled feature rows from TRAIN_START on, sorted by hour."""
    features, labels = store.read_features(), store.read_labels()
    if features.empty or labels.empty:
        raise ValueError("No stored features or labels to train on")
    features = features.assign(
        delivery_utc=pd.to_datetime(features["delivery_utc"], utc=True)
    )
    labels = labels.assign(
        delivery_utc=pd.to_datetime(labels["delivery_utc"], utc=True)
    )
    data = features.merge(
        labels[["delivery_utc", "y_neg"]], on="delivery_utc", how="inner"
    )
    data = data[data["y_neg"].notna() & (data["delivery_utc"] >= TRAIN_START)]
    data = data.assign(
        y_neg=data["y_neg"].astype("int64"),
        delivery_day=data["delivery_day"].astype(str),
    )
    return data.sort_values("delivery_utc").reset_index(drop=True)


def _days(data: pd.DataFrame, first: str, end: str) -> pd.DataFrame:
    """Rows with first <= delivery_day < end (ISO strings sort as dates)."""
    return data[(data["delivery_day"] >= first) & (data["delivery_day"] < end)]


def out_of_time_split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": _days(data, "0000-00-00", OOT_VALIDATION_START),
        "validation": _days(data, OOT_VALIDATION_START, OOT_TEST_START),
        "test": _days(data, OOT_TEST_START, OOT_TEST_END),
    }


def rolling_split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the newest labelled days into train, validation and holdout."""
    end = date.fromisoformat(data["delivery_day"].max()) + timedelta(days=1)
    holdout_start = end - timedelta(days=HOLDOUT_DAYS)
    validation_start = holdout_start - timedelta(days=VALIDATION_DAYS)
    train_start = validation_start - timedelta(days=TRAIN_DAYS)
    iso = date.isoformat
    return {
        "train": _days(data, iso(train_start), iso(validation_start)),
        "validation": _days(data, iso(validation_start), iso(holdout_start)),
        "holdout": _days(data, iso(holdout_start), iso(end)),
    }


def _window(rows: pd.DataFrame) -> tuple[str, str]:
    if rows.empty:
        return ("", "")
    return (rows["delivery_day"].min(), rows["delivery_day"].max())


# Scores


def scores(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Return PR-AUC, F1, precision and recall; PR-AUC is NaN without positives."""
    positives = int(y.sum())
    return {
        "pr_auc": float(average_precision_score(y, proba)) if positives else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "positives": float(positives),
        "rows": float(len(y)),
    }


def predictions(model: NegHourLGBM, rows: pd.DataFrame) -> dict[str, tuple]:
    """Return (score, yes or no) per row for LightGBM and each baseline."""
    proba = model.predict_proba(rows)
    result = {"lgbm": (proba, (proba >= model.threshold_).astype("int64"))}
    for baseline in BASELINES:
        result[baseline.name] = (baseline.predict_proba(rows), baseline.predict(rows))
    return result


def score_models(model: NegHourLGBM, rows: pd.DataFrame) -> dict[str, dict]:
    y = rows["y_neg"].to_numpy()
    return {
        name: scores(y, proba, pred)
        for name, (proba, pred) in predictions(model, rows).items()
    }


def segment_labels(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """Return season, day or night, and weekday or weekend per row."""
    hour = rows["hour_of_day"].astype(int)
    return {
        "season": rows["month"].astype(int).map(SEASONS),
        "daypart": np.where(hour.isin(DAY_HOURS), "day", "night"),
        "weekpart": np.where(rows["is_weekend"].astype(bool), "weekend", "weekday"),
    }


def segment_table(model: NegHourLGBM, rows: pd.DataFrame) -> pd.DataFrame:
    """Return PR-AUC and recall per segment for LightGBM and both baselines.

    A segment with fewer than MIN_SEGMENT_POSITIVES negative hours is listed
    with its counts, and its scores are left empty (README monitoring plan).
    """
    y = rows["y_neg"].to_numpy()
    preds = predictions(model, rows)
    table = []
    for kind, labels in segment_labels(rows).items():
        labels = np.asarray(labels)
        for value in sorted(set(labels)):
            mask = labels == value
            positives = int(y[mask].sum())
            entry: dict[str, object] = {
                "segment": kind,
                "value": value,
                "rows": int(mask.sum()),
                "positives": positives,
                "rate": float(y[mask].mean()),
            }
            for name, (proba, pred) in preds.items():
                scored = scores(y[mask], proba[mask], pred[mask])
                enough = positives >= MIN_SEGMENT_POSITIVES
                entry[f"{name}_pr_auc"] = scored["pr_auc"] if enough else np.nan
                entry[f"{name}_recall"] = scored["recall"] if enough else np.nan
            table.append(entry)
    return pd.DataFrame(table)


def calibration_table(proba: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Return mean predicted probability against the observed rate per bin."""
    bins = pd.cut(proba, CALIBRATION_BINS, include_lowest=True)
    frame = pd.DataFrame({"bin": bins, "p": proba, "y": y})
    table = frame.groupby("bin", observed=True).agg(
        rows=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")
    )
    return table.reset_index().assign(bin=lambda t: t["bin"].astype(str))


def calibration_summary(proba: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "mean_predicted": float(proba.mean()),
        "observed_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, proba)),
    }


# Promotion


def promotion_decision(metrics: dict[str, dict[str, float]]) -> tuple[bool, str]:
    """Apply the success criteria to holdout scores."""
    lgbm = metrics["lgbm"]
    best_baseline = max(metrics[b.name]["f1"] for b in BASELINES)
    if lgbm["positives"] < MIN_HOLDOUT_POSITIVES:
        return False, (
            f"only {int(lgbm['positives'])} negative hours in the holdout, "
            f"need {MIN_HOLDOUT_POSITIVES}; the current model stays"
        )
    if not lgbm["pr_auc"] >= config.TARGET_PR_AUC:
        return False, f"PR-AUC {lgbm['pr_auc']:.3f} < {config.TARGET_PR_AUC}"
    if lgbm["f1"] < best_baseline + config.TARGET_F1_MARGIN:
        return False, (
            f"F1 {lgbm['f1']:.3f} < best baseline {best_baseline:.3f} "
            f"+ {config.TARGET_F1_MARGIN}"
        )
    return True, (
        f"PR-AUC {lgbm['pr_auc']:.3f} and F1 {lgbm['f1']:.3f} "
        f"(best baseline {best_baseline:.3f}) meet the criteria"
    )


# Run


def run(
    mode: str = "retrain",
    store: FeatureStore | None = None,
    registry: ModelRegistry | None = None,
    tracker: Tracker | None = None,
    params: dict[str, object] | None = None,
) -> TrainingResult:
    """Train, score and, in retrain mode, promote. *registry* None = no promotion."""
    if mode not in ("retrain", "evaluate"):
        raise ValueError(f"Unknown mode: {mode}")
    data = load_dataset(store if store is not None else get_store())
    parts = rolling_split(data) if mode == "retrain" else out_of_time_split(data)
    score_part = "holdout" if mode == "retrain" else "test"
    for name, rows in parts.items():
        if rows.empty:
            raise ValueError(f"The {name} window has no labelled rows")

    train, validation, scored = parts["train"], parts["validation"], parts[score_part]
    model = NegHourLGBM(params=params).fit(
        train, train["y_neg"], validation, validation["y_neg"]
    )
    metrics = {f"validation/{k}": v for k, v in score_models(model, validation).items()}
    metrics.update(
        {f"{score_part}/{k}": v for k, v in score_models(model, scored).items()}
    )
    proba, y = model.predict_proba(scored), scored["y_neg"].to_numpy()

    result = TrainingResult(
        mode=mode,
        windows={name: _window(rows) for name, rows in parts.items()},
        metrics=metrics,
        segments=segment_table(model, scored),
        calibration=calibration_table(proba, y),
        importance=model.feature_importance(),
        threshold=model.threshold_,
        extra=calibration_summary(proba, y)
        | {"scale_pos_weight": float(model.scale_pos_weight_ or 0.0),
           "best_iteration": float(model.model_.best_iteration_ or 0)},
    )

    if mode == "retrain":
        holdout = {k.split("/", 1)[1]: v for k, v in metrics.items()
                   if k.startswith("holdout/")}
        result.promoted, result.reason = promotion_decision(holdout)
        if result.promoted and registry is not None:
            result.model_version = registry.register(
                model, registry_metrics(result), describe(result)
            )
        elif result.promoted:
            result.reason += " (dry run: not registered)"
            result.promoted = False
    else:
        result.reason = "evaluate mode never registers a model"

    if tracker is not None:
        result.run_id = tracker.log(
            run_name=f"{mode}-{result.windows[score_part][1]}",
            params=run_params(result, model),
            metrics=flat_metrics(result),
            tables={
                "segments": result.segments,
                "calibration": result.calibration,
                "feature_importance": result.importance,
            },
        )
    return result


def registry_metrics(result: TrainingResult) -> dict[str, float]:
    lgbm = result.metrics["holdout/lgbm"]
    best = max(result.metrics[f"holdout/{b.name}"]["f1"] for b in BASELINES)
    return {
        "pr_auc": lgbm["pr_auc"],
        "f1": lgbm["f1"],
        "recall": lgbm["recall"],
        "precision": lgbm["precision"],
        "best_baseline_f1": best,
        "threshold": result.threshold,
    }


def describe(result: TrainingResult) -> str:
    train, holdout = result.windows["train"], result.windows["holdout"]
    return (
        f"LightGBM trained {train[0]} to {train[1]}, holdout {holdout[0]} to "
        f"{holdout[1]}, threshold {result.threshold:.2f}"
    )


def run_params(result: TrainingResult, model: NegHourLGBM) -> dict[str, object]:
    params: dict[str, object] = {"mode": result.mode, "threshold": result.threshold}
    for name, (first, last) in result.windows.items():
        params[f"{name}_window"] = f"{first}..{last}"
    params.update({f"lgbm_{k}": v for k, v in model.params.items()})
    return params


def flat_metrics(result: TrainingResult) -> dict[str, float]:
    flat = {
        f"{part}/{name}/{metric}": value
        for key, values in result.metrics.items()
        for part, name in [key.split("/", 1)]
        for metric, value in values.items()
    }
    flat.update({f"calibration/{k}": v for k, v in result.extra.items()})
    flat["promoted"] = float(result.promoted)
    return flat


def summary_lines(result: TrainingResult) -> list[str]:
    """Return a plain-text report of one run."""
    part = "holdout" if result.mode == "retrain" else "test"
    lines = [f"## Training ({result.mode})", ""]
    for name, (first, last) in result.windows.items():
        lines.append(f"- {name}: {first} to {last}")
    lines += ["", f"Scores on the {part} window:", ""]
    for key, values in result.metrics.items():
        if key.startswith(f"{part}/"):
            lines.append(
                f"- {key.split('/', 1)[1]}: PR-AUC {values['pr_auc']:.3f}, "
                f"F1 {values['f1']:.3f}, precision {values['precision']:.3f}, "
                f"recall {values['recall']:.3f}, negative hours "
                f"{int(values['positives'])} of {int(values['rows'])}"
            )
    extra = result.extra
    lines += [
        "",
        f"Threshold {result.threshold:.2f}. Mean predicted probability "
        f"{extra['mean_predicted']:.3f} against an observed rate "
        f"{extra['observed_rate']:.3f}; Brier score {extra['brier']:.4f}.",
        "",
        f"Promotion: {'yes' if result.promoted else 'no'}, {result.reason}.",
    ]
    if result.model_version is not None:
        lines.append(
            f"Registered as {config.MODEL_NAME} version {result.model_version}."
        )
    return lines


def write_step_summary(result: TrainingResult) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines(result)) + "\n\n")
        table = result.segments.round(3).to_string(index=False)
        handle.write(f"```\n{table}\n```\n")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrain", "evaluate"], default="retrain")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read stored data and report, but log nothing and register nothing",
    )
    args = parser.parse_args(argv)

    registry = None if args.dry_run else get_registry()
    tracker: Tracker = NullTracker() if args.dry_run else MlflowTracker()
    result = run(args.mode, registry=registry, tracker=tracker)
    print("\n".join(summary_lines(result)))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(result.segments.round(3).to_string(index=False))
        print(result.calibration.round(3).to_string(index=False))
    write_step_summary(result)


if __name__ == "__main__":
    main()
