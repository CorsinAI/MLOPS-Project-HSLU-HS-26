# MS2 decision log

Decisions and deviations from the MS2 plan, newest last. Each entry says what
changed and why, so `docs/ms2_summary.pdf` can point here.

| # | Date | Decision | Reason |
|---|------|----------|--------|
| 1 | 2026-11-04 | One git branch per phase, merged by PR with a merge commit. | Keeps small commits (E4) and runs CI before `main`. Schedules run from `main`. |
| 2 | 2026-11-04 | Review PDFs are not in the repo; the owner deleted them locally. | Anonymous reviews must not land in a public repo. |
| 3 | 2026-11-04 | No Hugging Face Parquet fallback. Hopsworks is the only planned store. | Owner decision. The storage interface stays, so a backend can still be swapped. |
| 4 | 2026-11-04 | Labels table gets `n_neg_quarters` (NaN before the 15-minute switch). | Lets the "any quarter < 0" label be tested later without rebuilding (E1). |
