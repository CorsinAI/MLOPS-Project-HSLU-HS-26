"""Round-trip check for the Hopsworks project. Prints results, never the key.

Creates a temporary feature group, inserts rows, upserts one changed row,
reads the result back and deletes the group.
"""

from __future__ import annotations

import sys

import hopsworks
import pandas as pd

from neghour import config

GROUP_NAME = "neghour_conn_check"


def main() -> int:
    if not config.HOPSWORKS_API_KEY:
        print("FAIL key: HOPSWORKS_API_KEY is not set")
        return 1

    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT, api_key_value=config.HOPSWORKS_API_KEY
    )
    print(f"PASS login: project {project.name}")

    store = project.get_feature_store()
    group = store.get_or_create_feature_group(
        name=GROUP_NAME,
        version=1,
        primary_key=["id"],
        event_time="ts",
        online_enabled=False,
        description="Temporary NegativeHours connectivity check",
    )
    times = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    rows = pd.DataFrame({"id": [1, 2, 3], "ts": times, "value": [1.0, 2.0, 3.0]})
    try:
        group.insert(rows, write_options={"wait_for_job": True})
        print("PASS insert: 3 rows")

        changed = rows.iloc[[0]].assign(value=9.0)
        group.insert(changed, write_options={"wait_for_job": True})
        result = group.read().sort_values("id")
        print(f"PASS read: {len(result)} rows, values {result['value'].tolist()}")

        if result["value"].tolist() != [9.0, 2.0, 3.0]:
            print("FAIL upsert: expected [9.0, 2.0, 3.0]")
            return 1
        print("PASS upsert: replaced in place, no duplicate")
    finally:
        group.delete()
        print("PASS cleanup: temporary group deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
