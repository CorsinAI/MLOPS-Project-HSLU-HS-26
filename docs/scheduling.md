# Scheduling with Google Cloud Scheduler

GitHub cron starts this repo's jobs hours late (decision #47). Google Cloud
Scheduler fires on time and in the Europe/Berlin time zone, so the clock change
needs no second cron line. It does not run any code itself: at the set time it
sends one HTTP request to GitHub, which starts the workflow by
`workflow_dispatch`. A run started this way normally begins within a minute or
two.

Cost: 3 jobs per billing account are free; each further job costs $0.10 per
month. A billing account is required.

## The three jobs

| Job | Berlin time | Starts | Why |
|---|---|---|---|
| `neghour-inference` | 10:05 daily | `inference.yml` | Forecast after the cut-off, before 12:00 |
| `neghour-labels` | 13:30 daily | `daily_pipeline.yml` | Prices for D+1 are published at 13:00 |
| `neghour-training` | Sunday 03:00 | `weekly_training.yml` | Weekly retraining |

The GitHub cron of `daily_pipeline.yml` stays on as a backup for features and
labels. Inference rebuilds the day's features itself, so it does not depend on
the morning cron runs.

## Setup (about 15 minutes)

1. **GitHub token.** GitHub → Settings → Developer settings → Fine-grained
   tokens → Generate. Repository access: only
   `CorsinAI/MLOPS-Project-HSLU-HS-26`. Permissions: Actions, read and write.
   Pick an expiry date after the exam and note it; jobs fail once it expires.
2. **Google Cloud.** Create or pick a project, link the billing account, and
   enable the Cloud Scheduler API.
3. **Create the jobs.** In Cloud Shell, set the token once and run the three
   commands. `europe-west6` is Zurich; any region works.

```bash
TOKEN=github_pat_...   # the token from step 1
REPO=https://api.github.com/repos/CorsinAI/MLOPS-Project-HSLU-HS-26/actions/workflows
HEADERS="Accept=application/vnd.github+json,Content-Type=application/json,X-GitHub-Api-Version=2022-11-28,Authorization=Bearer $TOKEN"

gcloud scheduler jobs create http neghour-inference \
  --location=europe-west6 --schedule="5 10 * * *" --time-zone="Europe/Berlin" \
  --uri="$REPO/inference.yml/dispatches" --http-method=POST \
  --headers="$HEADERS" --message-body='{"ref":"main"}' \
  --max-retry-attempts=3 --min-backoff=60s

gcloud scheduler jobs create http neghour-labels \
  --location=europe-west6 --schedule="30 13 * * *" --time-zone="Europe/Berlin" \
  --uri="$REPO/daily_pipeline.yml/dispatches" --http-method=POST \
  --headers="$HEADERS" --message-body='{"ref":"main"}' \
  --max-retry-attempts=3 --min-backoff=60s

gcloud scheduler jobs create http neghour-training \
  --location=europe-west6 --schedule="0 3 * * 0" --time-zone="Europe/Berlin" \
  --uri="$REPO/weekly_training.yml/dispatches" --http-method=POST \
  --headers="$HEADERS" \
  --message-body='{"ref":"main","inputs":{"mode":"retrain"}}' \
  --max-retry-attempts=3 --min-backoff=60s
```

4. **Test.** Run one job by hand and check that GitHub started the workflow:

```bash
gcloud scheduler jobs run neghour-labels --location=europe-west6
gh run list --workflow daily_pipeline.yml --limit 3
```

GitHub answers a good request with HTTP 204. A 401 or 403 means the token is
wrong or lacks the Actions permission; a 404 means the workflow file is not on
`main` yet.

## Notes

- The token is stored in the job's headers, which project members can read.
  It can only start workflows in this one repository.
- Timing is checked twice: the inference pipeline refuses to start before
  10:00 Berlin on D-1 and marks a prediction made after 12:00 as late.
  `python -m scripts.schedule_report --workflow inference.yml --event
  workflow_dispatch` reports the real start times.
- Apache Airflow remains an option for later, for example on a small VM paid
  from the $50 credit. It would replace these jobs, not the pipelines: the
  pipelines are plain Python entry points and run the same under any trigger.
