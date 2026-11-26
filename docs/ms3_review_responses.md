# MS2 review responses: working draft for `docs/ms3_summary`

Draft written with Claude Code on 2026-11-26 from the editorial summary
(20.11.2026) and reviews A and B. It lists what was done per point, with a
status, commits and files, in editorial order. The owner writes the final
`docs/ms3_summary.md`; entries marked "Owner" need the owner's own words.
Decisions are in `docs/ms3_decisions.md`.

Order of work: the editorial asks to settle A1 before the first training run and
E1 before inference is scheduled, so A1 and E1 came first, then training (with
E8), then inference.

## Must

**A1 · Stored rows are rewritten for a week after their cut-off (also B1).**
A stored feature row for day D is now frozen at gate closure, 12:00 Berlin on
D-1, and never rewritten after that, complete or not. Gate closure is the
reference because inference rebuilds the row between 10:00 and 12:00 and
predicts from it. Each row stores `fetched_at`, the time its sources were
downloaded, so a row rebuilt later (for example by a backfill) is visible as
such. Labels are still refreshed. Measured: stored version 1 against a fresh
reload for 2026-10-01 to 2026-11-18 differs in 1 of 1,177 hours (0.1 %).
Status: Changed in code. Live once features version 2 is backfilled (one write of
about 25,500 rows, not run yet). Commits ad1389a, 9125671. Files:
`src/neghour/pipelines/feature_pipeline.py`, `scripts/compare_reload.py`,
decisions #45, #46.

**E1 · A forecast before 12:00 needs a start time you control (also A2, B2).**
Measured from the GitHub run history: of 83 scheduled runs (2026-10-02 to
2026-11-26), 22 % started before 10:00 Berlin, 5 % between 10:00 and 12:00, and
10 % of days had a run in that window. Inference therefore has no cron: it runs
by `workflow_dispatch`, refuses to start before the cut-off, stores
`predicted_at` and `on_time`, and a late run never replaces an on-time
prediction. The feature step summary logs each run's real start time.
Scheduler chosen: Google Cloud Scheduler calls inference at 10:05 Berlin, the
label refresh at 13:30 and retraining on Sundays (decision #60). It still has
to be set up in Google Cloud (`docs/scheduling.md`).
Status: Partly, Changed once the jobs run. Commits 9125671, cc0da98. Files:
`scripts/schedule_report.py`, `src/neghour/pipelines/inference_pipeline.py`,
`.github/workflows/inference.yml`, `docs/scheduling.md`, decisions #47, #48, #60.

**E8 · Seasonality (MS1 E2).** Training reports PR-AUC and recall per season,
day and night, weekday and weekend, for LightGBM and both baselines, plus a
calibration table. On the test year (Jul 2025 to Jun 2026), LightGBM recall is
0.76 to 0.91 in spring, summer and autumn against 0.37 to 0.60 for
NaiveYesterday. Winter has 9 negative hours and night 36, so winter is counted,
not scored. Probabilities run too high: mean 0.077 against an observed rate of
0.054.
Status: Changed. Commit 89022a9. Files:
`src/neghour/pipelines/training_pipeline.py`, decisions #50 to #52.

## Should

**B3 · Runs can overlap.** The feature workflows share the concurrency group
`feature-writes` and never cancel a running job.
Status: Changed. Commit 63857f5. File: `.github/workflows/daily_pipeline.yml`.

**E2 · Make failures visible and retry more than HTTP 5xx.** Login, reads and
writes retry on HTTP 5xx, dropped connections, timeouts and storage I/O errors
(decision #43). A failed scheduled feature run, and any failed inference or
training run, opens a GitHub issue. The README shows badges.
Status: Changed. Commits 22b1e2e, 63857f5. Files:
`src/neghour/storage/hopsworks_store.py`, `.github/workflows/*.yml`, `README.md`.

**E3 · Measure what the conservative forecast vintage costs.**
Status: Not yet.

**E4 · The residual-load proxy drifts.** Capacities stay fixed (#42), and the
drift is named as a monitoring signal.
Status: Changed. File: `README.md`, "Monitoring plan", decision #57.

**E5 · Work in smaller steps and own the design.** Owner.

## Could

**A3 · No ms2 tag (also B4).** Owner: `git tag -a ms2 8d0e0c6 -m "MS2 hand-in"`
and `git push origin ms2`.
Status: Not yet.

**E6 · Stubs.** Training and inference are built. `app/app.py` and the empty
notebook remain.
Status: Partly. Commits 89022a9, cc0da98.

**E7 · Originality line.** Owner: one README sentence, once the owner has
checked the four KTH ID2223 projects the editors named.
Status: Not yet.
