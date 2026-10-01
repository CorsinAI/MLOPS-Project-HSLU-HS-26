"""Tests for feature assembly and leakage validation."""

# TODO: test build_day returns exactly 24 rows for any valid delivery_date
# TODO: test build_day columns match EXPECTED_COLUMNS from validation module
# TODO: test validate_no_leakage raises LeakageError when available_at >= D-1 10:00 Berlin
# TODO: test validate_no_leakage passes for available_at strictly before cut-off
# TODO: test validate_schema raises ValueError listing missing columns
# TODO: test is_holiday flag is correct for German public holidays (e.g. Christmas)
# TODO: test residual_load_proxy is derived without using post-cutoff data
