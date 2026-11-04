# MS2 decision log

Decisions and deviations from the MS2 plan, newest last. Each entry says what
changed and why, so `docs/ms2_summary.pdf` can point here.

| # | Date | Decision | Reason |
|---|------|----------|--------|
| 1 | 2026-11-04 | One git branch per phase, merged by PR with a merge commit. | Keeps small commits (E4) and runs CI before `main`. Schedules run from `main`. |
| 2 | 2026-11-04 | Review PDFs are not in the repo; the owner deleted them locally. | Anonymous reviews must not land in a public repo. |
| 3 | 2026-11-04 | No Hugging Face Parquet fallback. Hopsworks is the only planned store. | Owner decision. The storage interface stays, so a backend can still be swapped. |
| 4 | 2026-11-04 | Labels table gets `n_neg_quarters` (NaN before the 15-minute switch). | Lets the "any quarter < 0" label be tested later without rebuilding (E1). |
| 5 | 2026-11-04 | Direct dependencies are pinned with `==` as well as locked in `uv.lock`; `train` and `app` extras hold packages MS2 does not need. | Reproducible builds and a smaller image. DagsHub code is untouched. |
| 6 | 2026-11-04 | The Hopsworks check runs as a manual GitHub workflow, not locally, and also tests upsert behaviour. | Needs only the GitHub secret, and proves the behaviour the labels/features split relies on. |
| 7 | 2026-11-04 | Residual-load proxy uses fixed documented capacities, not fitted coefficients. | Nothing is fitted, so nothing can leak from later periods. Tree models need the ranking, not exact scale. |
| 8 | 2026-11-04 | Hopsworks key is `delivery_epoch` (UTC hour as integer seconds), added inside the backend; `delivery_utc` is the event time. | Same UTC-hour key as decided, in a type that is safe as a primary key. Frames elsewhere stay unchanged. |
| 9 | 2026-11-04 | `validate_no_leakage` runs from the first scheduled pipeline, ahead of Phase 4. | The editorial asks for the cut-off check in the first scheduled run (A3). |
| 10 | 2026-11-04 | The inference job is removed from `daily_pipeline.yml` until the model exists. | It raises `NotImplementedError`, so every daily run would fail despite good features. |
| 11 | 2026-11-04 | `neg_hours_last_7d` stays as a feature. | In the proposal, and Review B named it a strength. |
| 12 | 2026-11-04 | `load_h_d7` added as a feature. | The proxy needs expected demand for the same weekday; D-2 can be a different kind of day. |
| 13 | 2026-11-04 | Missing data is NaN plus a flag, never carried forward. | A copied forecast or price would be invented data that looks real. |
| 14 | 2026-11-04 | During a source outage the daily run writes only the newest delivery day. Backfills fail instead. | The daily run rebuilds 8 days, so an outage could otherwise overwrite good rows with NaN. |
| 15 | 2026-11-04 | Holiday weights use population by state for 2025 (census 2022), statistikportal.de. Luxembourg is left out. | Official figures; the table sums to the published 83,467,117. Luxembourg is about 0.8 % of the zone. |
| 16 | 2026-11-04 | `is_holiday` is 1 only when all 16 states are off; `holiday_share` carries regional holidays. | Keeps the binary flag meaningful next to the share (B2). |
| 17 | 2026-11-04 | Feature group stays at version 1; no version bump. | Nothing was written before the schema was final, so the bump in the plan is unnecessary. |
| 18 | 2026-11-04 | A leakage test failed to catch a removed weather filter, so late-publishing sources now have their own tests. | Found by deliberately breaking the code. All filters are now shown to be covered. |
