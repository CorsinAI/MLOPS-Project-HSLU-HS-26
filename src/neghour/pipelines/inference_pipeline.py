"""
Inference pipeline — runs daily at 10:05 Berlin time on D-1, after the feature pipeline.

Steps:
  1. Load feature rows for delivery day D from the Hopsworks feature group.
  2. Load the registered production model from Hopsworks Model Registry.
  3. Predict P(price < 0) and binary label for each of the 24 delivery hours.
  4. Write predictions to the Hopsworks predictions feature group.

Post-evaluation (separate step, run after D-1 12:45 when actuals are published):
  5. Fetch actual DE-LU prices from SMARD for D.
  6. Derive y_neg actuals and append to prediction rows in the feature group.
  7. Metrics are then visible in the Streamlit dashboard.

Triggered by: .github/workflows/daily_pipeline.yml (step 2 of 2)
Invoked via:  python -m neghour.pipelines.inference_pipeline
"""

from __future__ import annotations

from datetime import date, timedelta

# TODO: implement run(delivery_date: date | None = None) -> None
#       default: delivery_date = date.today() + timedelta(days=1)
# TODO: implement post_evaluate(delivery_date: date) -> None
#       called the next day to attach actual labels; could be a separate workflow step


def run(delivery_date: date | None = None) -> None:
    """Load model and write 24-hour predictions for *delivery_date*."""
    if delivery_date is None:
        delivery_date = date.today() + timedelta(days=1)

    # TODO: implement inference pipeline steps
    raise NotImplementedError


def post_evaluate(delivery_date: date) -> None:
    """Attach actual y_neg labels to prediction rows after prices are published."""
    # TODO: implement post-evaluation step
    raise NotImplementedError


if __name__ == "__main__":
    run()
