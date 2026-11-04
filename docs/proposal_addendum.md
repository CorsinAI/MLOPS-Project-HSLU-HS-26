# Proposal addendum (MS2)

The submitted proposal stays unchanged in `docs/proposal.pdf`. This page records
where the project now differs from it, and why.

| Topic | Proposal says | Now | Why | Review |
|---|---|---|---|---|
| Task | P(price < 0) "+ price"; Figure 1 shows a regressor | Classification only | The decision is yes or no per hour. A second target added confusion. | E2a-i |
| Label | Price < 0 for hour h | Hourly mean price < 0 from SMARD filter 4169; `n_neg_quarters` also stored | The market clears 15-minute products since 01.10.2025. The hourly mean is the one definition valid across the whole dataset. | E1 |
| Auction | "One price per hour" | 15-minute products since 01.10.2025 | Market change | E1 |
| Times | Mostly no timezone | Europe/Berlin everywhere; storage in UTC | Removes ambiguity, including around clock changes | A2 |
| Run time | Daily run at 10:00 | Cron at 01:17 and 05:17 UTC; 10:00 is only the data cut-off | A job starting at 10:00 cannot finish by 10:00, and cron started 4–9 h late | B1 |
| Weather history | Previous Runs "from Jan 2024" | Forecasts begin 2024-01-20, and 100 m wind 2024-02-17; earlier rows stay NaN | Checked against the API | A1 |
| Weather vintage | History 2 days ahead, live latest | `_previous_day2` in both | Same forecast age in training and serving. `_previous_day1` would use post-cut-off runs for later hours. | A1 |
| Weather location | Not stated | Four points, aggregated per variable | Germany's weather differs by region, and so do its wind and solar parks | E2a-ii |
| Publication time | 12:45 | 13:00 rule | SMARD has no per-row timestamp, so the rule stays conservative | A3 |
| Storage | One Hopsworks store | Separate feature and label groups | Labels refresh daily; features stay as they were at the cut-off | |
| Features | Listed | Adds `load_h_d7`, cyclical encodings, `is_weekend`, `holiday_share` | Proxy needs same-weekday demand; seasonality and regional holidays | E2a-iii, B2 |
| Imbalance | "Extra weight" | `scale_pos_weight` in MS3; no upsampling | Upsampling would create training-serving skew | E2a-iv |
| Serving | Hourly chart | Unchanged for now | Later milestone | E3 |

Details and smaller decisions are in `docs/ms2_decisions.md`.
