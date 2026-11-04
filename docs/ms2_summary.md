# NegativeHours: Milestone 2 summary

Corsin Hidber · HSLU MLOps HS26 · Repo: `CorsinAI/MLOPS-Project-HSLU-HS-26`

## 1. Summary of the milestone

### What a reviewer should look at first

1. `tests/test_leakage.py`: changes or removes data published after the cut-off and requires identical rows.
2. `src/neghour/availability.py`: all publication rules in one place.
3. `scripts/check_store_quality.py`, and its output in backfill run 41915601205.
4. `docs/proposal_addendum.md`: every change from the proposal, with its review point.

### What works now

The feature pipeline runs end to end. GitHub Actions fetches SMARD data and Open-Meteo forecasts, builds each row as known at 10:00 Berlin on D-1, validates it and upserts it to Hopsworks. Live runs and backfills share the same code. Training and inference remain out of scope.

| Proof | Result | GitHub Actions run |
|---|---|---|
| Hopsworks round trip | Login, insert, upsert in place, read and delete all passed | 41913408675 |
| Daily pipeline, manual start | Success: 216 feature and 216 label rows, no failed source | 41914046157 |
| Scheduled daily run | Cron is live on `main`; first slots are 01:17, 05:17 and 12:17 UTC on 05.11. Not yet proven | Pending |
| Full backfill and stored-data report | Success in 243 s; all checks passed, and again on rerun | 41915601205, 41921336517 |
| Pipeline inside the Docker image | Success: 216 feature and 216 label rows | 41920910156 |
| CI on `main` | Lock check, ruff, 136 tests and Docker build passed | 41920897685 |

**Stored data, read back from Hopsworks after the backfill:**

- **Rows:** 24,960 in `neghour_features` and 24,960 in `neghour_labels`, covering 2024-01-01 to 2026-11-05, with 0 duplicate UTC hours.
- **Clock changes:** spring days 2024-03-31, 2025-03-30 and 2026-03-29 have 23 rows. Autumn days 2024-10-27, 2025-10-26 and 2026-10-25 have 25.
- **Leakage:** no feature uses the label or prices of day D. Price features end at D-1, and `tests/test_leakage.py` corrupts day-D prices and requires identical rows. In storage, 0 rows have any `available_at_*` after the cut-off, which the report recomputes from `delivery_day`. The closest value is 3 hours before it.
- **Gaps:** `wind_100m` and the proxy are empty for 1,135 hours, to 2024-02-17 05:00 UTC. Irradiance, cloud cover and temperature are empty for 469 hours, to 2024-01-20 11:00 UTC. Both match the start of archived forecasts. Price and actuals have no gaps. `n_neg_quarters` starts at the 15-minute switch, as designed.
- **Negative hours:** 457 of 8,784 in 2024 (5.20 %), 573 of 8,760 in 2025 (6.54 %), 462 of 7,416 in 2026 so far (6.23 %). The total is 1,492 (5.98 %), identical to the dry run.
- **Hopsworks jobs:** 0 executions. Two `compute_stats` job definitions exist but have never run, because statistics are off and inserts write Delta tables directly.

### Key decisions

- **Label:** `y_neg` = hourly mean price < 0 from SMARD filter 4169. It is the one definition valid across the whole dataset. `n_neg_quarters` is stored so a quarter-hour label can be tested later.
- **Vintage:** `_previous_day2` forecasts in both backfill and live, so training and serving see forecasts of the same age.
- **Time:** storage in UTC, keyed by UTC hour; delivery days and the cut-off in Europe/Berlin. Rows are selected by timestamp, so a late cron run writes the same features.
- **Availability:** rule-based, because SMARD publishes no per-row timestamp.
- **Late data:** NaN plus a `missing_*` flag, never carried forward.

**Known limits:** SMARD revisions and the stability of `_previous_day2` values between runs are not measured. The 6-hour publication margins are assumptions. One container run (41920298746) failed on a Hopsworks lock timeout, so writes now retry on server errors; that retry has only been tested with fakes. All decisions are in `docs/ms2_decisions.md`.

## 2. Response to reviews

The editorial uses E2 twice, so its two uses appear here as E2a and E2b.

| ID | Point | Status | Answer | Commit or file |
|---|---|---|---|---|
| A1 | Same weather vintage in backfill and live | Changed | `_previous_day2` in both: `_previous_day1` would use runs after the cut-off for later hours. Availability is T − 48 h + 6 h. | 304cb25, `sources/open_meteo.py` |
| A2 | Europe/Berlin everywhere; CET/CEST in UTC cron | Changed | Europe/Berlin stated throughout; storage in UTC. The cut-off is computed in Berlin time, so the UTC cron shift cannot move it. 23 and 25 hour days tested and confirmed in storage. | 49ebba4, b9f0ddc, `timeutils.py` |
| A3 | Test that fails on data after the cut-off | Changed | Checked before each write, by tests, and on the full stored table: 0 late rows. | 245c48b, aa75bc7, 689eb2f |
| B1 | Start before 10:00; define late and missing data | Changed | Morning cron at 01:17 and 05:17 UTC, since runs started 4–9 h late; 12:17 UTC refreshes labels. The 10:00 data cut-off stays. A missing or late input gives NaN plus a flag and a log line, not the last value, which would look like real data. | c91f563, dc8bdbf, 94863b8 |
| B2 | Weighted holiday share | Changed | `holiday_share` by state population, next to `is_holiday`. | c0d414d, `features/calendar.py` |
| E1 | Hourly label after the 15-minute switch | Changed | Checked: filter 4169 stays hourly and returns the mean of the four quarters after the switch; `smard.py` reads both resolutions. A battery operator acts per quarter, but quarter prices do not exist before the switch, so only the hourly mean gives one label across train, validation and test. Since the switch 510 hours are negative by mean, 626 by any quarter, 408 by all four, so the label misses 116 actionable hours. `n_neg_quarters` lets MS3 test the any-quarter label without a rebuild. | 6603e83, 6f812e5, 85bbfe8 |
| E2a | Resolve the confusing proposal points | Changed | Addendum lists each change; the submitted proposal is untouched. | 5d9084c, `proposal_addendum.md` |
| E2a-i | Classification or price prediction | Changed | Classification only: the decision is yes or no per hour. | 6f812e5, `proposal_addendum.md` |
| E2a-ii | Region covered by each input | Changed | Prices and load are zone-wide (DE-LU). Weather varies, so four points, averaged per variable; reasoning in the README. | 304cb25, 72e3906 |
| E2a-iii | Seasonality within year, day and week | Partly | Features for all three cycles: day, week and year sine and cosine pairs plus `is_weekend`, argued in the README. Modelling follows in MS3. | c0d414d, `README.md` |
| E2a-iv | Upsampling causes training-serving skew | Partly | Upsampling rejected in writing. `scale_pos_weight` follows in MS3. | 021b736, `README.md` |
| E2b | Originality against KTH SE3 regressors | Partly | README states the difference: rare-event classification for DE-LU, not a price level. The statement that the check was done is not added yet. | 021b736, `README.md` |
| E3 | Rolling PR-AUC/F1 and drift check | Partly | Plan written: rolling PR-AUC and F1 against both baselines, and a drift check. Not built, because it needs a model. | `README.md`, Monitoring plan |
| E4 | Commit in small steps | Changed | Over 30 focused commits, one pull request per phase. | PRs 1, 2, 4–10, `git log` |
