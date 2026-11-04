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
