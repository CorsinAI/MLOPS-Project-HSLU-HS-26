"""Tests for source modules. All use recorded fixtures, never live calls."""

import json
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from neghour.sources import http, open_meteo, smard

FIXTURES = Path(__file__).parent / "fixtures" / "smard"
WEATHER_FIXTURES = Path(__file__).parent / "fixtures" / "open_meteo"
PRICE_FILTER = 4169


def fixture_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name.startswith("index_"):
            name = f"{PRICE_FILTER}_DE-LU_{name}"
        path = FIXTURES / name
        if not path.exists():
            return httpx.Response(404)
        return httpx.Response(200, json=json.loads(path.read_text(encoding="utf-8")))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_prices_stay_hourly_across_the_quarter_hour_switch() -> None:
    prices = smard.fetch_prices(
        date(2025, 9, 29), date(2025, 10, 5), client=fixture_client()
    )
    assert len(prices) == 168
    assert prices["delivery_utc"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


@pytest.mark.parametrize(
    ("day", "expected"),
    [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25)],
)
def test_prices_cover_short_and_long_days(day: date, expected: int) -> None:
    prices = smard.fetch_prices(day, day, client=fixture_client())
    assert len(prices) == expected
    assert prices["delivery_utc"].is_unique


def test_prices_have_rule_based_availability() -> None:
    day = date(2026, 10, 25)
    prices = smard.fetch_prices(day, day, client=fixture_client())
    assert (prices["available_at"] == pd.Timestamp("2026-10-24 11:00", tz="UTC")).all()


def test_fetch_keeps_only_the_requested_days() -> None:
    day = date(2026, 3, 29)
    prices = smard.fetch_prices(day, day, client=fixture_client())
    assert prices["delivery_utc"].min() == pd.Timestamp("2026-03-28 23:00", tz="UTC")
    assert prices["delivery_utc"].max() == pd.Timestamp("2026-03-29 21:00", tz="UTC")


def test_http_failures_raise_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http, "RETRY_DELAY_S", 0)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(http.SourceUnavailableError):
        smard.fetch_prices(date(2026, 1, 1), date(2026, 1, 1), client=client)
    assert len(calls) == http.ATTEMPTS


def weather_client(requests: list[httpx.Request] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        start = request.url.params["start_date"]
        path = WEATHER_FIXTURES / f"previous_runs_{start}.json"
        return httpx.Response(200, json=json.loads(path.read_text(encoding="utf-8")))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_weather_request_uses_day2_for_all_points_and_no_model_run() -> None:
    requests: list[httpx.Request] = []
    day = date(2026, 10, 25)
    open_meteo.fetch_weather(day, day, client=weather_client(requests))
    params = requests[0].url.params
    assert all(name.endswith("_previous_day2") for name in params["hourly"].split(","))
    assert len(params["latitude"].split(",")) == 4
    assert params["wind_speed_unit"] == "ms"
    assert "model_run" not in params


def test_weather_covers_the_25_hour_day() -> None:
    day = date(2026, 10, 25)
    weather = open_meteo.fetch_weather(day, day, client=weather_client())
    assert len(weather) == 25
    assert weather[open_meteo.WEATHER_COLUMNS].notna().all().all()


def test_weather_availability_is_42_hours_before_valid_time() -> None:
    day = date(2026, 10, 25)
    weather = open_meteo.fetch_weather(day, day, client=weather_client())
    delta = weather["delivery_utc"] - weather["available_at_weather"]
    assert (delta == pd.Timedelta(hours=42)).all()


def test_early_2024_weather_stays_nan_without_availability() -> None:
    day = date(2024, 1, 10)
    weather = open_meteo.fetch_weather(day, day, client=weather_client())
    assert len(weather) == 24
    assert weather[open_meteo.WEATHER_COLUMNS].isna().all().all()
    assert weather["available_at_weather"].isna().all()
