"""The schedule report measures real start times in Berlin time."""

from __future__ import annotations

from scripts.schedule_report import report, start_table

RUNS = [
    # started 09:38 UTC = 10:38 Berlin in winter: inside the 10-12 window
    {"databaseId": 1, "createdAt": "2026-11-26T09:37:00Z",
     "startedAt": "2026-11-26T09:38:00Z", "conclusion": "success"},
    # started 08:03 UTC = 09:03 Berlin: before the cut-off
    {"databaseId": 2, "createdAt": "2026-11-22T08:03:00Z",
     "startedAt": "2026-11-22T08:03:00Z", "conclusion": "success"},
    # started 15:36 UTC = 16:36 Berlin, same day as run 1
    {"databaseId": 3, "createdAt": "2026-11-26T15:36:00Z",
     "startedAt": "2026-11-26T15:36:00Z", "conclusion": "failure"},
]


def test_start_times_are_classified_in_berlin_time() -> None:
    table = start_table(RUNS).set_index("run_id")
    assert table.loc[1, "in_window"] and not table.loc[1, "before_cutoff"]
    assert table.loc[2, "before_cutoff"] and not table.loc[2, "in_window"]
    assert not table.loc[3, "before_cutoff"] and not table.loc[3, "in_window"]
    assert table.loc[1, "berlin_hour"] == 10


def test_report_gives_shares_of_runs_and_of_days() -> None:
    lines = report(start_table(RUNS))
    assert "Runs started before 10:00 Berlin: 33%" in lines
    assert "Days with a run 10:00-12:00: 50%" in lines
    assert lines[0].startswith("Scheduled runs: 3 on 2 days")
