"""
Inference pipeline. Not built yet (MS3).

Planned steps:
  1. Load the feature rows for delivery day D from Hopsworks.
  2. Load the registered model from the Hopsworks Model Registry.
  3. Predict P(price < 0) and yes or no for every delivery hour of D. A day has
     23, 24 or 25 hours.
  4. Write predictions, keyed by delivery_utc, to the predictions feature group.

Outcomes are not copied here. Scores join predictions to the labels group on
delivery_utc, as in the README monitoring plan.

The forecast must be out before the auction closes at 12:00 Berlin. GitHub cron
cannot guarantee that, so MS3 needs a scheduler with better control of the start
time.

Invoked via:  python -m neghour.pipelines.inference_pipeline
"""

from __future__ import annotations

from datetime import date, timedelta

from neghour.timeutils import berlin_today


def run(delivery_date: date | None = None) -> None:
    """Write predictions for *delivery_date*, by default Berlin tomorrow."""
    if delivery_date is None:
        delivery_date = berlin_today() + timedelta(days=1)

    # TODO: implement inference pipeline steps
    raise NotImplementedError


if __name__ == "__main__":
    run()
