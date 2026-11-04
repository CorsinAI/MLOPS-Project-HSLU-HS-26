"""Record small real API responses into tests/fixtures. Run by hand when needed."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from neghour.config import (
    SMARD_BASE_URL,
    SMARD_LOAD_FILTER,
    SMARD_PRICE_FILTER,
    SMARD_REGION,
    SMARD_SOLAR_FILTER,
    SMARD_WIND_OFFSHORE_FILTER,
    SMARD_WIND_ONSHORE_FILTER,
)
from neghour.sources import open_meteo
from neghour.sources.http import get_json, new_client

FIXTURES = Path("tests/fixtures/smard")
WEATHER_FIXTURES = Path("tests/fixtures/open_meteo")
WEEKS = {
    "15-minute switch": "2025-10-01",
    "spring DST": "2026-03-29",
    "autumn DST": "2026-10-25",
}


def record_smard(filter_id: int, resolution: str = "hour") -> None:
    client = new_client()
    base = f"{SMARD_BASE_URL}/{filter_id}/{SMARD_REGION}"
    starts = get_json(client, f"{base}/index_{resolution}.json")["timestamps"]

    kept: list[int] = []
    for day in WEEKS.values():
        moment = pd.Timestamp(day, tz="UTC").value // 1_000_000
        chunk = max(start for start in starts if start <= moment)
        kept.append(chunk)
        name = f"{filter_id}_{SMARD_REGION}_{resolution}_{chunk}.json"
        body = get_json(client, f"{base}/{name}")
        (FIXTURES / name).write_text(json.dumps(body), encoding="utf-8")

    index_name = f"{filter_id}_{SMARD_REGION}_index_{resolution}.json"
    index_body = json.dumps({"timestamps": kept})
    (FIXTURES / index_name).write_text(index_body, encoding="utf-8")


def record_open_meteo() -> None:
    """Save the raw bodies that fetch_weather requests for two test days."""

    class Recorder(httpx.Client):
        def get(self, url: str, **kwargs: object) -> httpx.Response:
            response = super().get(url, **kwargs)
            start = dict(kwargs["params"])["start_date"]
            path = WEATHER_FIXTURES / f"previous_runs_{start}.json"
            path.write_text(response.text, encoding="utf-8")
            return response

    for day in (date(2024, 1, 10), date(2026, 10, 25)):
        open_meteo.fetch_weather(day, day, client=Recorder(timeout=60))


if __name__ == "__main__":
    FIXTURES.mkdir(parents=True, exist_ok=True)
    WEATHER_FIXTURES.mkdir(parents=True, exist_ok=True)
    record_smard(SMARD_PRICE_FILTER)
    record_smard(SMARD_PRICE_FILTER, "quarterhour")
    record_smard(SMARD_LOAD_FILTER)
    record_smard(SMARD_SOLAR_FILTER)
    record_smard(SMARD_WIND_ONSHORE_FILTER)
    record_smard(SMARD_WIND_OFFSHORE_FILTER)
    record_open_meteo()
