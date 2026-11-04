"""Record small real API responses into tests/fixtures. Run by hand when needed."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neghour.config import SMARD_BASE_URL, SMARD_PRICE_FILTER, SMARD_REGION
from neghour.sources.http import get_json, new_client

FIXTURES = Path("tests/fixtures/smard")
WEEKS = {
    "15-minute switch": "2025-10-01",
    "spring DST": "2026-03-29",
    "autumn DST": "2026-10-25",
}


def record_smard(filter_id: int) -> None:
    client = new_client()
    base = f"{SMARD_BASE_URL}/{filter_id}/{SMARD_REGION}"
    starts = get_json(client, f"{base}/index_hour.json")["timestamps"]

    kept: list[int] = []
    for name, day in WEEKS.items():
        moment = pd.Timestamp(day, tz="UTC").value // 1_000_000
        chunk = max(start for start in starts if start <= moment)
        kept.append(chunk)
        body = get_json(client, f"{base}/{filter_id}_{SMARD_REGION}_hour_{chunk}.json")
        name_on_disk = f"{filter_id}_{SMARD_REGION}_hour_{chunk}.json"
        (FIXTURES / name_on_disk).write_text(json.dumps(body), encoding="utf-8")
        print(f"recorded {name}: {name_on_disk}")

    index_name = f"{filter_id}_{SMARD_REGION}_index_hour.json"
    index_body = json.dumps({"timestamps": kept})
    (FIXTURES / index_name).write_text(index_body, encoding="utf-8")


if __name__ == "__main__":
    FIXTURES.mkdir(parents=True, exist_ok=True)
    record_smard(SMARD_PRICE_FILTER)
