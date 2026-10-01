"""
Training pipeline — runs weekly (every Sunday at 02:00 UTC).

Steps:
  1. Read all feature rows from the Hopsworks feature group.
  2. Split out-of-time: train 2024 · validate H1 2025 · test Jul 2025–Jun 2026.
  3. Fit baselines: NaiveYesterday and NaiveLastWeek.
  4. Fit LightGBM with scale_pos_weight for the rare positive class (~3–6%).
  5. Evaluate all models: PR-AUC, F1@best-threshold, month-by-month breakdown.
  6. Log experiment (params, metrics, artifacts) to MLflow on DagsHub.
  7. Register model in Hopsworks Model Registry if:
       PR-AUC >= 0.75  AND  F1(lgbm) >= max(F1(baselines)) + 0.10

Triggered by: .github/workflows/weekly_training.yml
Invoked via:  python -m neghour.pipelines.training_pipeline
"""

from __future__ import annotations

# TODO: implement run() -> None
# TODO: implement _split(df) -> tuple[df_train, df_val, df_test]
# TODO: implement _evaluate(model, X, y) -> dict  (pr_auc, f1, monthly_f1)
# TODO: implement _promote_if_criteria_met(run_id, metrics) -> bool
# TODO: add confusion matrix and PR-curve plots as MLflow artifacts


def run() -> None:
    """Retrain all models and promote LightGBM if success criteria are met."""
    # TODO: implement training pipeline steps
    raise NotImplementedError


if __name__ == "__main__":
    run()
