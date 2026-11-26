"""The reload comparison counts changed values and treats two NaN as equal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.compare_reload import differences, report


def frame(values: list[float], naive: bool = False) -> pd.DataFrame:
    hours = pd.date_range("2026-10-01", periods=len(values), freq="h", tz="UTC")
    if naive:
        hours = hours.tz_localize(None)
    return pd.DataFrame({"delivery_utc": hours, "wind_d2": values})


def test_changed_and_new_nan_values_count_as_differences() -> None:
    stored = frame([1.0, np.nan, 3.0, np.nan])
    reloaded = frame([1.0, np.nan, 3.5, 4.0])
    diff = differences(stored, reloaded, ["wind_d2"])
    assert diff["wind_d2"].tolist() == [False, False, True, True]


def test_hours_are_matched_even_when_storage_drops_the_timezone() -> None:
    diff = differences(frame([1.0, 2.0], naive=True), frame([1.0, 2.0]), ["wind_d2"])
    assert len(diff) == 2 and not diff["wind_d2"].any()


def test_report_gives_the_share_of_changed_hours() -> None:
    diff = differences(frame([1.0, 2.0, 3.0, 4.0]), frame([1.0, 2.0, 3.0, 9.0]),
                       ["wind_d2"])
    lines = report(diff, ["wind_d2"])
    assert "Hours with any difference: 1 (25.0%)" in lines
    assert "  wind_d2: 1 (25.0%)" in lines
