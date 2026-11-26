"""Report when scheduled runs really started (review E1).

Reads the scheduled runs of a workflow from the GitHub API through the `gh`
CLI and reports, in Berlin time, the share of runs and of days with a run that
started before 10:00 (the cut-off) and between 10:00 and 12:00 (the window an
inference run needs). GitHub does not record which cron slot fired a run, and
runs start up to about 12 hours late, so runs are not matched to slots.

Usage:
  python scripts/schedule_report.py                    # daily_pipeline.yml
  python scripts/schedule_report.py --input runs.json  # saved `gh run list` JSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import time
from pathlib import Path

import pandas as pd

from neghour.config import BERLIN_TZ, CUTOFF_HOUR, GATE_CLOSURE_HOUR

FIELDS = "databaseId,createdAt,startedAt,conclusion,event"


def fetch_runs(workflow: str, limit: int) -> list[dict]:
    output = subprocess.run(
        ["gh", "run", "list", "--workflow", workflow, "--event", "schedule",
         "--limit", str(limit), "--json", FIELDS],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(output)


def start_table(runs: list[dict]) -> pd.DataFrame:
    """Return one row per run with its real start time in UTC and Berlin."""
    rows = []
    for run in runs:
        started = pd.Timestamp(run.get("startedAt") or run["createdAt"])
        berlin = started.tz_convert(BERLIN_TZ)
        rows.append(
            {
                "run_id": run["databaseId"],
                "started_utc": started,
                "berlin_day": berlin.date(),
                "berlin_hour": berlin.hour,
                "before_cutoff": berlin.time() < time(CUTOFF_HOUR),
                "in_window": time(CUTOFF_HOUR) <= berlin.time()
                < time(GATE_CLOSURE_HOUR),
                "conclusion": run.get("conclusion", ""),
            }
        )
    return pd.DataFrame(rows)


def report(table: pd.DataFrame) -> list[str]:
    """Return report lines: shares of runs and of days, and starts per hour."""
    if table.empty:
        return ["No scheduled runs found."]
    days = table.groupby("berlin_day")[["before_cutoff", "in_window"]].any()
    hours = table["berlin_hour"].value_counts().sort_index()
    return [
        f"Scheduled runs: {len(table)} on {len(days)} days, "
        f"{days.index.min()} to {days.index.max()} (Berlin)",
        f"Runs started before 10:00 Berlin: {table['before_cutoff'].mean():.0%}",
        f"Runs started 10:00-12:00 Berlin: {table['in_window'].mean():.0%}",
        f"Days with a run before 10:00: {days['before_cutoff'].mean():.0%}",
        f"Days with a run 10:00-12:00: {days['in_window'].mean():.0%}",
        "Starts per Berlin hour: "
        + ", ".join(f"{hour:02d}h {count}" for hour, count in hours.items()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default="daily_pipeline.yml")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()

    if args.input:
        runs = json.loads(args.input.read_text("utf-8"))
    else:
        runs = fetch_runs(args.workflow, args.limit)
    runs = [r for r in runs if r.get("event", "schedule") == "schedule"]
    table = start_table(runs)
    print("\n".join(report(table)))


if __name__ == "__main__":
    main()
