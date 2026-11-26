# NegativeHours

[![Daily pipeline](https://github.com/CorsinAI/MLOPS-Project-HSLU-HS-26/actions/workflows/daily_pipeline.yml/badge.svg)](https://github.com/CorsinAI/MLOPS-Project-HSLU-HS-26/actions/workflows/daily_pipeline.yml)
[![CI](https://github.com/CorsinAI/MLOPS-Project-HSLU-HS-26/actions/workflows/ci.yml/badge.svg)](https://github.com/CorsinAI/MLOPS-Project-HSLU-HS-26/actions/workflows/ci.yml)

> Predict which hours of tomorrow will have a **negative day-ahead electricity
> price** in the DE-LU market, from data known at 10:00 Europe/Berlin, two hours
> before the auction gate closes.

**Author:** Corsin Hidber · HSLU MLOps HS26
**Target:** PR-AUC ≥ 0.75 · F1 ≥ best baseline + 0.10 (proposal). Promotion gate for
now: PR-AUC ≥ 0.60 · F1 ≥ best baseline + 0.05 (decision #58).
**Scope:** DE-LU bidding zone · hourly · 2024 to live
**Status:** Milestone 3 in progress. Feature, training and inference pipelines are
built; inference is triggered by Google Cloud Scheduler (decision #60, `docs/scheduling.md`).

Decisions and changes are logged in [`docs/ms2_decisions.md`](docs/ms2_decisions.md)
(#1 to #44) and [`docs/ms3_decisions.md`](docs/ms3_decisions.md) (#45 on). Changes to the original proposal
are in [`docs/proposal_addendum.md`](docs/proposal_addendum.md).

**Originality:** this is rare-event classification of negative hours for DE-LU;
the decision is yes or no per hour before the auction closes. I looked for the KTH
SE3 price regressors named in the MS1 editorial and could not identify them exactly.
The closest work I found is Vanherpe (2026), a KTH thesis I checked from its
abstract, which studies how weather forecast errors affect intraday prices in SE3.

---

## Task

Binary classification. For each delivery hour:

`y_neg = 1` if the hourly mean DE-LU day-ahead price is below 0 €/MWh.

The market has cleared 15-minute products since 01.10.2025. SMARD's hourly series
stays continuous across that switch: the hourly price before it, the mean of the
four quarter hours after it. The label therefore stays closest to the pre-switch
hourly price across train, validation and test. Separate quarter prices only start
at the switch, so an "any quarter" label would raise the share of negative hours at
that date, and training and test would measure different things. A battery operator
acts per quarter hour, so `n_neg_quarters` is stored as well, which lets that label
be tested later without rebuilding. SMARD rounds to two decimals, so a mean between
−0.005 and 0 reads as 0.00 and counts as not negative.

## Architecture

```mermaid
flowchart TD
    S1[SMARD<br/>prices, load, solar, wind] --> FP
    S2[Open-Meteo Previous Runs<br/>4 points, day-2 forecasts] --> FP
    S3[Calendar and holidays] --> FP
    FP[Feature pipeline<br/>fetch, build as of cut-off, validate, upsert] --> FS
    FS[(Hopsworks feature store<br/>features, labels, predictions)]
    FS --> TP[Training pipeline<br/>fit, score against baselines, promote]
    TP --> MR[(Hopsworks Model Registry)]
    TP --> ML[MLflow on DagsHub<br/>runs and metrics]
    MR --> IP[Inference pipeline<br/>refresh day D, predict, store]
    FS --> IP
    IP --> FS
    GH[GitHub Actions<br/>daily cron, backfill, manual training] --> FP
    GH --> TP
    TR[Trigger at about 10:05 Berlin<br/>not chosen yet] --> IP
```

Live runs and backfills use the same code. The only difference is the date range.

## Time rules

- **Every stored timestamp is UTC.** The key is the UTC hour of delivery, never a
  local date and hour, because German days have 23, 24 or 25 hours.
- **Delivery days and the cut-off are defined in Europe/Berlin.** All times in this
  repo are Berlin time unless marked UTC.
- **"Today" and "tomorrow" come from the Berlin calendar**, not the runner clock.
- **Cut-off:** a row for delivery day D may only use data with
  `available_at <= 10:00 Berlin on D-1`.

Rows are selected by timestamp, never by when a job runs, so a late cron run writes
the same features as an on-time one.

SMARD has no per-row publication time, so `available_at` is rule-based. The rules
are in [`src/neghour/availability.py`](src/neghour/availability.py):

| Data | Counts as available | Basis |
|---|---|---|
| Day-ahead price for day X | 13:00 Berlin on X-1 | Conservative after the 12:00 auction |
| SMARD actual for hour T | End of T + 6 h | Conservative; newest values were 1–3 h old when checked |
| `_previous_day2` forecast, valid T | T − 48 h + 6 h | 48 h is the documented offset; 6 h is an assumed margin |
| Calendar and holidays | 00:00 Berlin on D-1 | Known in advance |

The cut-off is enforced before every write and tested in
[`tests/test_leakage.py`](tests/test_leakage.py). Those tests also remove or corrupt
everything published after the cut-off and require the row to stay identical.

## Features

| Group | Columns | Why |
|---|---|---|
| Price | `price_h_d1`, `price_h_d7`, `neg_hours_last_7d` | Recent price level. Negative hours cluster in high-renewable periods. |
| Weather forecast | `irradiance`, `cloud_cover`, `wind_100m`, `temperature` | Expected solar, wind and heating or cooling demand. |
| Actuals | `load_d2`, `solar_d2`, `wind_d2`, `load_h_d7` | Latest measured system state. D-7 has the same weekday as D. |
| Proxy | `residual_load_proxy` | Expected demand minus expected wind and solar. Low values mean a surplus. |
| Calendar | `hour_of_day`, `weekday`, `month`, `is_weekend`, sine and cosine pairs | Seasonality, below. |
| Holidays | `is_holiday`, `holiday_share` | Demand falls on holidays, many of which are regional. |

Each source also has an `available_at_*` column and a `missing_*` flag.

### Spatial coverage

Prices, production and load cover the whole bidding zone. Weather does not, so it
uses four fixed points, averaged per variable:

| Point | Location | Used for |
|---|---|---|
| `n_coast` | 54.4, 8.7 | Wind: most German capacity is in the windy north |
| `ne` | 53.0, 13.0 | Wind and solar: the north-eastern wind belt and large solar parks |
| `centre` | 51.2, 10.5 | Wind and solar: a mid-country reference |
| `south` | 48.6, 11.0 | Solar: the south has the strongest irradiance and much rooftop solar |

- **Wind** averages `n_coast`, `ne`, `centre`.
- **Irradiance and cloud cover** average `ne`, `centre`, `south`.
- **Temperature** averages all four, because it drives demand everywhere.

### Seasonality

Negative prices follow three cycles, and each is an explicit feature:

- **Within a day:** the solar peak at noon, none at night (`hour_sin`, `hour_cos`).
- **Within a week:** lower demand at weekends (`weekday_sin`, `weekday_cos`,
  `is_weekend`).
- **Within a year:** more solar in spring and summer (`year_sin`, `year_cos`).

Sine and cosine pairs let a model see that hour 23 is next to hour 0.

### Holidays

`holiday_share` is the share of Germany's population with a public holiday that
day, by state. Epiphany, for example, only applies in three states, so a binary flag
would miss it. `is_holiday` is 1 only when all 16 states are off. Luxembourg is about
0.8 % of the zone and is left out. Holidays that apply to only part of a state, such
as Assumption Day in much of Bavaria, are not counted.

### Class imbalance

About 6 % of hours are negative. MS3 will use `scale_pos_weight`, not upsampling.
Upsampling would change the class balance the model sees in training but not in
serving, which is training-serving skew.

## Late and missing data

- A source that still fails after three attempts gives NaN plus a `missing_*` flag,
  a warning, and a line in the GitHub step summary.
- Values are never carried forward. A copied forecast or price would be invented
  data that looks real.
- A stored feature row for day D is frozen at gate closure, 12:00 Berlin on D-1,
  and is never rewritten after that, complete or not (decision #45). Training
  therefore sees the row the forecast used. A day that was never stored can still
  be written later; its `fetched_at` after gate closure marks it as rebuilt.
  Labels are refreshed by every run.
- During an outage the daily run writes only the newest delivery day, and never over
  a complete stored row. Labels are skipped if quarter counts are missing. The next
  run repairs what is missing.
- Hopsworks login, reads and writes retry on server errors, dropped connections and
  storage I/O errors, because upserts are safe to repeat.
- The run fails if validation or storage fails, or if SMARD and Open-Meteo are both
  down. A backfill fails when a source is down.

**Backfill coverage:** rows start at 2024-01-01, but archived forecasts start later.
Solar, cloud and temperature forecasts begin on 2024-01-20 12:00 UTC, and 100 m wind
on 2024-02-17 06:00 UTC. Earlier hours stay as NaN with `missing_weather` set, and
MS3 training drops them.

**Known limitations:** SMARD actuals may be revised after first publication, so
backfilled rows can be slightly cleaner than live ones. Measured on 1,177 stored hours
(2026-10-01 to 2026-11-18): 1 hour differs from a fresh reload (decision #46).
`available_at` comes from rules, so the leakage checks prove the rules were followed;
`fetched_at` records when a row's data was really downloaded.

## Data sources and licences

| Source | Used for | Key | Licence |
|---|---|---|---|
| [SMARD](https://www.smard.de) (Bundesnetzagentur) | Prices (filter 4169), load, solar, wind | None | CC BY 4.0 |
| [Open-Meteo](https://open-meteo.com) Previous Runs API | `_previous_day2` forecasts, default model | None | CC BY 4.0; free tier is non-commercial |
| [Statistikportal](https://www.statistikportal.de/de/bevoelkerung/flaeche-und-bevoelkerung) | Population by state, 2025, based on census 2022 | n/a | See source |

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). A
[Hopsworks](https://www.hopsworks.ai) project is needed to store data.

```bash
uv sync --frozen --extra dev --extra train
cp .env.example .env   # add HOPSWORKS_API_KEY to .env; never commit that file
```

Writing to Hopsworks needs its Delta library, which the lockfile installs only on
Linux and on Apple-silicon macOS. On Windows, reads and dry runs work; run writes
through the GitHub workflows or WSL.

## Run

```bash
# Daily run: Berlin today - 7 days through Berlin tomorrow
uv run --frozen python -m neghour.pipelines.feature_pipeline

# Backfill: defaults to 2024-01-01 through tomorrow
uv run --frozen python -m neghour.pipelines.backfill
uv run --frozen python -m neghour.pipelines.backfill --first-day 2025-01-01 --last-day 2025-01-31

# Dry run against live sources, storing nothing
STORAGE_BACKEND=memory uv run --frozen python -m neghour.pipelines.feature_pipeline

# Report on what is stored
uv run --frozen python scripts/check_store_quality.py

# Training (needs the train extra). evaluate = test year, never registers
uv run --frozen python -m neghour.pipelines.training_pipeline --mode retrain
uv run --frozen python -m neghour.pipelines.training_pipeline --mode evaluate --dry-run

# Inference for Berlin tomorrow; refused before 10:00 Berlin on D-1
uv run --frozen python -m neghour.pipelines.inference_pipeline
uv run --frozen python -m neghour.pipelines.inference_pipeline --date 2026-11-27 --dry-run

# Measurements for the MS2 review
uv run --frozen python -m scripts.compare_reload --first-day 2026-10-01 --last-day 2026-11-18
uv run --frozen python -m scripts.schedule_report
```

`--dry-run` reads stored data and writes nothing. Both pipelines are idempotent: rows are upserted by UTC delivery hour, so a rerun
adds nothing.

### Docker

```bash
docker build -t neghour .
docker run --rm --env-file .env neghour
```

Keys are passed at run time and are never part of the image.

### Tests

```bash
uv run --frozen pytest -v
uv run --frozen ruff check src tests scripts app
```

Source tests use recorded API responses in `tests/fixtures`, never live calls.
Refresh them with `uv run --frozen python scripts/record_fixtures.py`.

## Orchestration

| Workflow | Trigger | Steps |
|---|---|---|
| `daily_pipeline.yml` | Cron 01:17, 05:17 and 12:17 UTC, or manual | Feature pipeline |
| `inference.yml` | Cloud Scheduler 10:05 Berlin, or manual | Refresh day D, predict, store |
| `weekly_training.yml` | Cloud Scheduler Sunday 03:00 Berlin, or manual | Retrain or evaluate |
| `backfill.yml` | Manual | Backfill, then a report on stored data |
| `ci.yml` | Push to main, PR, or manual | Lock check, ruff, pytest, Docker build |
| `check_hopsworks.yml` | Manual | Hopsworks round-trip check |
| `docker_run.yml` | Manual | Builds the image and runs the pipeline inside it |

GitHub cron is in UTC and starts this repo's jobs hours late. Of 83 scheduled runs
between 2026-10-02 and 2026-11-26, 22 % started before 10:00 Berlin and 5 % between
10:00 and 12:00 (decision #47, `scripts/schedule_report.py`). Feature values are
selected by timestamp, so a late run computes the same values. Since decision #45 a
run after 12:00 Berlin no longer rewrites tomorrow's stored row; it only fills a
day that was never stored, and refreshes labels.

A third slot at 12:17 UTC (13:17 CET, 14:17 CEST) refreshes labels. The auction
closes at 12:00, so the morning runs cannot see the next day's prices. This run
stores those labels the same day, not the next morning.

**Inference timing:** the forecast must start after 10:00 and be out before 12:00
Berlin, which cron cannot promise. `inference.yml` therefore has no cron and runs by
`workflow_dispatch`. The pipeline refuses to start before the cut-off, stores
`predicted_at` and `on_time`, and a late run never replaces an on-time prediction.
Google Cloud Scheduler calls it at 10:05 Berlin, starts a label refresh at 13:30
and retraining on Sundays (decision #60). Setup: `docs/scheduling.md`. MLflow tracks
runs and models; it does not schedule.

**Overlaps and failures:** feature writers share one concurrency group and never
cancel a running job. A failed scheduled feature run, and any failed inference or
training run, opens a GitHub issue.

**Secrets:** `HOPSWORKS_API_KEY`, and `DAGSHUB_TOKEN` for MLflow on DagsHub (not set
yet; without it runs are logged locally). Locally they go in `.env`, which is
gitignored.

**Variable:** `HOPSWORKS_HOST` in GitHub Variables, the Hopsworks instance host. It
is not a secret and is not hardcoded, so the code runs against any instance.

**Hopsworks cost:** the feature groups are offline only, with `stream=False`, Delta
and statistics off, so the client writes the tables directly and inserts start no
Spark job. Reads go through the Feature Query Service. Creating the groups also
created two `compute_stats` job definitions; neither has run.

## Monitoring plan (MS3)

Not built yet: predictions are stored from the first inference run on. The plan:

- **Rolling scores:** PR-AUC and F1 over the last 30 delivery days, joining
  predictions to labels on the UTC hour, for the model and for both baselines
  (same hour on D-1 and on D-7). Negative hours are rare and seasonal, so a
  window with few of them is reported as such, not as a score.
- **Drift check:** each week, compare `residual_load_proxy`, `wind_100m` and
  `irradiance` with the training window using the population stability index,
  and warn above 0.2.
- **Capacity drift:** the proxy uses wind and solar capacity fixed at end-2024
  values, while installed capacity grows. A rising PSI of `residual_load_proxy`
  is the expected sign; it is watched, not corrected (decision #57).
- **Late forecasts:** the share of delivery days without an on-time prediction
  (`on_time` in the predictions group).
- **Calibration:** mean predicted probability against the observed rate. On the
  test year the probabilities ran too high (decision #52).

## Repository

```text
src/neghour/
├── availability.py   publication rules, in one place
├── timeutils.py      Berlin days, cut-off, DST-safe lags
├── config.py         settings, grid points, populations
├── sources/          SMARD, Open-Meteo
├── features/         build, calendar, validation
├── storage/          interface, Hopsworks, in-memory
├── models/           baselines, LightGBM wrapper, model registry
├── tracking.py       MLflow logging
└── pipelines/        feature, backfill, training, inference

scripts/              store report, reload comparison, schedule report
app/app.py            dashboard placeholder until MS3
```
