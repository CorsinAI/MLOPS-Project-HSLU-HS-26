"""Report on what is actually stored. Exits non-zero if the data is wrong.

Every check reads the stored tables, so it also proves the round trip through
storage. The cut-off is recomputed from `delivery_day` instead of trusting the
stored `cutoff_utc`. Also lists the Hopsworks jobs and their executions.
"""

from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from neghour import config
from neghour.features.validation import AVAILABILITY_COLUMNS
from neghour.storage import get_store
from neghour.timeutils import cutoff_utc, delivery_hours

# Clock-change days and the number of rows each one must have.
CLOCK_CHANGE_DAYS = {
    "2024-03-31": 23,
    "2024-10-27": 25,
    "2025-03-30": 23,
    "2025-10-26": 25,
    "2026-03-29": 23,
    "2026-10-25": 25,
}

# Archived forecasts start here (README, "Backfill coverage").
WEATHER_START = pd.Timestamp("2024-01-20 12:00", tz="UTC")
WIND_START = pd.Timestamp("2024-02-17 06:00", tz="UTC")
QUARTER_START = pd.Timestamp("2025-09-30 22:00", tz="UTC")

# Columns allowed to be empty, and the hour from which they must be filled.
FEATURES_EMPTY_UNTIL = {
    "irradiance": WEATHER_START,
    "cloud_cover": WEATHER_START,
    "temperature": WEATHER_START,
    "wind_100m": WIND_START,
    "residual_load_proxy": WIND_START,
    "available_at_weather": WIND_START,
}
LABELS_EMPTY_UNTIL = {"n_neg_quarters": QUARTER_START}


def with_utc_times(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame with each time column as UTC, however storage returned it."""
    times = [c for c in frame if c.startswith(("available_at", "cutoff", "delivery_u"))]
    return frame.assign(**{c: pd.to_datetime(frame[c], utc=True) for c in times})


def check_rows(features: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    first = date.fromisoformat(str(features["delivery_day"].min()))
    last = date.fromisoformat(str(features["delivery_day"].max()))
    expected = len(delivery_hours(first, last))
    feature_duplicates = int(features["delivery_utc"].duplicated().sum())
    label_duplicates = int(labels["delivery_utc"].duplicated().sum())

    print("\n== Rows and keys ==")
    print(f"Delivery days: {first} to {last}")
    print(f"Feature rows: {len(features)} of {expected} expected hours")
    print(f"Label rows: {len(labels)}")
    print(f"Duplicate UTC hours in features: {feature_duplicates}")
    print(f"Duplicate UTC hours in labels: {label_duplicates}")

    failures = []
    if len(features) != expected:
        failures.append(f"features have {len(features)} rows, expected {expected}")
    if feature_duplicates or label_duplicates:
        failures.append("duplicate UTC hours")
    if set(features["delivery_utc"]) != set(labels["delivery_utc"]):
        failures.append("features and labels cover different hours")
    return failures


def check_day_lengths(features: pd.DataFrame) -> list[str]:
    print("\n== Clock-change days ==")
    sizes = features.groupby("delivery_day").size()
    failures = []
    for day, expected in CLOCK_CHANGE_DAYS.items():
        rows = int(sizes.get(day, 0))
        print(f"{day}: {rows} rows, expected {expected}")
        if rows != expected:
            failures.append(f"{day} has {rows} rows, expected {expected}")
    print(f"All day lengths: {sizes.value_counts().sort_index().to_dict()}")
    return failures


def check_missing(
    frame: pd.DataFrame, table: str, empty_until: dict[str, pd.Timestamp]
) -> list[str]:
    """Report NaN per column, and fail on any NaN outside its expected window."""
    print(f"\n== Missing values: {table} ==")
    failures = []
    for column in frame.columns:
        empty = frame.loc[frame[column].isna(), "delivery_utc"]
        if empty.empty:
            continue
        print(f"{column}: {len(empty)} empty, {empty.min()} to {empty.max()}")
        limit = empty_until.get(column)
        if limit is None or empty.max() >= limit:
            failures.append(f"{column} is empty outside its expected window")
    if not failures:
        print(f"Every other column of {len(frame.columns)} is complete")
    flags = frame.filter(like="missing_").astype(bool).sum().to_dict()
    if flags:
        print(f"Rows flagged: { {k: int(v) for k, v in flags.items()} }")
    return failures


def check_leakage(features: pd.DataFrame) -> list[str]:
    print("\n== Leakage: available_at <= 10:00 Berlin on D-1 ==")
    days = features["delivery_day"].astype(str)
    cutoffs = {day: cutoff_utc(date.fromisoformat(day)) for day in days.unique()}
    cutoff = pd.to_datetime(days.map(cutoffs), utc=True)

    failures = []
    if not (features["cutoff_utc"] == cutoff).all():
        failures.append("stored cutoff_utc differs from the recomputed cut-off")
    for column in AVAILABILITY_COLUMNS:
        late = int((features[column] > cutoff).sum())
        latest = (features[column] - cutoff).max()
        print(f"{column}: {late} rows late, closest to the cut-off {latest}")
        if late:
            failures.append(f"{late} rows have {column} after the cut-off")
    return failures


def report_negative_share(features: pd.DataFrame, labels: pd.DataFrame) -> None:
    print("\n== Negative-price hours per delivery year ==")
    years = labels[["delivery_utc", "y_neg"]].merge(
        features[["delivery_utc", "delivery_day"]], on="delivery_utc"
    )
    years["year"] = years["delivery_day"].astype(str).str[:4]
    years["y_neg"] = years["y_neg"].astype(int)
    for year, rows in years.groupby("year")["y_neg"]:
        print(f"{year}: {rows.sum()} of {len(rows)} hours, {rows.mean():.2%}")
    total = years["y_neg"]
    print(f"All: {total.sum()} of {len(total)} hours, {total.mean():.2%}")


def report_jobs() -> None:
    from neghour.storage.hopsworks_store import login

    print("\n== Hopsworks jobs and executions ==")
    jobs = login().get_job_api().get_jobs()
    print(f"Jobs in project: {len(jobs)}")
    for job in jobs:
        executions = job.get_executions()
        print(f"{job.name}: {len(executions)} executions")
        for execution in executions:
            print(
                f"  {execution.id} {execution.final_status} "
                f"{execution.submission_time} {execution.duration} ms"
            )


def run_checks(features: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    """Print the report and return what failed."""
    features = with_utc_times(features)
    labels = with_utc_times(labels)
    failures = check_rows(features, labels)
    failures += check_day_lengths(features)
    failures += check_missing(features, "features", FEATURES_EMPTY_UNTIL)
    failures += check_missing(labels, "labels", LABELS_EMPTY_UNTIL)
    failures += check_leakage(features)
    report_negative_share(features, labels)
    return failures


def main() -> int:
    store = get_store()
    failures = run_checks(store.read_features(), store.read_labels())
    if config.STORAGE_BACKEND == "hopsworks":
        report_jobs()

    print("\n== Result ==")
    for failure in failures:
        print(f"FAIL {failure}")
    print("PASS" if not failures else f"{len(failures)} checks failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
