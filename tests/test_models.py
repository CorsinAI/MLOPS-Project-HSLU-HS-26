"""Tests for baseline and LightGBM model wrappers."""

# TODO: test NaiveYesterday.predict matches same-hour label from D-1
# TODO: test NaiveLastWeek.predict matches same-hour label from D-7
# TODO: test both baselines return 0 (majority class) for missing look-back dates
# TODO: test NegHourLGBM fit/predict cycle on a small synthetic DataFrame
# TODO: test NegHourLGBM.predict_proba returns values in [0, 1]
# TODO: test threshold_from_validation returns a float in [0.1, 0.9]
# TODO: test scale_pos_weight is set correctly from training label distribution
