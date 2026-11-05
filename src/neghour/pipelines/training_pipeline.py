"""
Training pipeline. Not built yet (MS3). Its weekly schedule stays off until then.

Planned steps:
  1. Read features and labels from Hopsworks, joined on delivery_utc.
  2. Drop rows before 2024-02-17 06:00 UTC, where archived forecasts are missing.
     They stay stored.
  3. Fit baselines: NaiveYesterday and NaiveLastWeek.
  4. Fit LightGBM with scale_pos_weight for the rare positive class (about 6 %).
  5. Evaluate all models: PR-AUC, F1 at the best threshold, month by month.
  6. Log the experiment (params, metrics, artifacts) to MLflow on DagsHub.
  7. Register the model in the Hopsworks Model Registry if:
       PR-AUC >= 0.75  AND  F1(lgbm) >= max(F1(baselines)) + 0.10

Two kinds of run:
  - Out-of-time evaluation, once: train to the end of 2024, validate on H1 2025,
    test on Jul 2025-Jun 2026. The test year is not reused for model selection.
  - Weekly retraining: a rolling window of recent data, so new hours reach the
    model. Promotion compares against both baselines on recent data, never on the
    test year. The window length is set in MS3.

Triggered by: .github/workflows/weekly_training.yml
Invoked via:  python -m neghour.pipelines.training_pipeline
"""

from __future__ import annotations

# TODO: implement run() -> None
# TODO: implement _split(df) -> tuple[df_train, df_val, df_test]
# TODO: implement _rolling_window(df) -> tuple[df_train, df_val]
# TODO: implement _evaluate(model, X, y) -> dict  (pr_auc, f1, monthly_f1)
# TODO: implement _promote_if_criteria_met(run_id, metrics) -> bool
# TODO: add confusion matrix and PR-curve plots as MLflow artifacts


def run() -> None:
    """Retrain all models and promote LightGBM if success criteria are met."""
    # TODO: implement training pipeline steps
    raise NotImplementedError


if __name__ == "__main__":
    run()
