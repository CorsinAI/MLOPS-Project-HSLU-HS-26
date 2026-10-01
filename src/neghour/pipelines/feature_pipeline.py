"""
Feature pipeline — runs daily at 10:00 Berlin time on D-1.

Steps:
  1. Fetch prices, generation, and load from SMARD (Energy-Charts fallback).
  2. Fetch Open-Meteo weather forecast for delivery day D.
  3. Assemble 24 feature rows for D (one per delivery hour).
  4. Validate schema and enforce the 10:00 cut-off (no leakage).
  5. Upsert rows into the Hopsworks feature group.

Triggered by: .github/workflows/daily_pipeline.yml (step 1 of 2)
Invoked via:  python -m neghour.pipelines.feature_pipeline
"""

from __future__ import annotations

from datetime import date, timedelta

# TODO: implement run(delivery_date: date | None = None) -> None
#       default: delivery_date = date.today() + timedelta(days=1)  (tomorrow = D)
# TODO: wire sources.smard / sources.energy_charts → features.build → validation → hopsworks
# TODO: add structured logging: run_id, delivery_date, row_count, source_used, duration_s
# TODO: emit a GitHub Actions step summary with the 24-row preview


def run(delivery_date: date | None = None) -> None:
    """Build and store feature rows for *delivery_date* (default: tomorrow)."""
    if delivery_date is None:
        delivery_date = date.today() + timedelta(days=1)

    # TODO: implement pipeline steps
    raise NotImplementedError


if __name__ == "__main__":
    run()
