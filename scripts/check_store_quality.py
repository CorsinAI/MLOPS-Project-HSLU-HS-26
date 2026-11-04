"""Report on what is actually stored. Exits non-zero if the data is incomplete."""

from __future__ import annotations

import sys
from datetime import date

from neghour.storage import get_store
from neghour.timeutils import delivery_hours


def main() -> int:
    store = get_store()
    features = store.read_features()
    labels = store.read_labels()

    first = date.fromisoformat(features["delivery_day"].min())
    last = date.fromisoformat(features["delivery_day"].max())
    expected = len(delivery_hours(first, last))
    duplicates = int(features["delivery_utc"].duplicated().sum())
    missing_share = features.filter(like="missing_").mean().round(4).to_dict()

    print(f"Delivery days: {first} to {last}")
    print(f"Feature rows: {len(features)} of {expected} expected hours")
    print(f"Duplicate feature keys: {duplicates}")
    print(f"Label rows: {len(labels)}")
    print(f"Negative hours: {int(labels['y_neg'].sum())}")
    print(f"Share of rows with missing inputs: {missing_share}")

    return 0 if len(features) == expected and duplicates == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
