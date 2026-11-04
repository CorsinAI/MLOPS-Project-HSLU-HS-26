"""Login and group settings. The Hopsworks client itself is replaced by a fake."""

from __future__ import annotations

from typing import Any

import pytest

from neghour import config
from neghour.storage import hopsworks_store
from neghour.storage.hopsworks_store import GROUP_SETTINGS, HopsworksStore, login


class FakeStore:
    def __init__(self) -> None:
        self.group_settings: dict[str, Any] = {}

    def get_or_create_feature_group(self, **settings: Any) -> Any:
        self.group_settings = settings
        return self

    def insert(self, rows: Any, **options: Any) -> None:
        pass


class FakeProject:
    def __init__(self) -> None:
        self.store = FakeStore()

    def get_feature_store(self) -> FakeStore:
        return self.store


@pytest.fixture
def login_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_login(**settings: Any) -> FakeProject:
        calls.append(settings)
        return FakeProject()

    monkeypatch.setattr(hopsworks_store.hopsworks, "login", fake_login)
    return calls


@pytest.mark.parametrize(
    "value",
    [
        "eu-west.cloud.hopsworks.ai",
        "https://eu-west.cloud.hopsworks.ai",
        "https://eu-west.cloud.hopsworks.ai/",
    ],
)
def test_login_uses_the_configured_host(
    monkeypatch: pytest.MonkeyPatch, login_calls: list[dict[str, Any]], value: str
) -> None:
    monkeypatch.setattr(config, "HOPSWORKS_HOST", value)
    login()
    assert login_calls[0]["host"] == "eu-west.cloud.hopsworks.ai"


def test_a_missing_host_fails_instead_of_using_the_public_default(
    monkeypatch: pytest.MonkeyPatch, login_calls: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(config, "HOPSWORKS_HOST", "")
    with pytest.raises(RuntimeError, match="HOPSWORKS_HOST"):
        login()
    assert login_calls == []


def test_groups_are_created_with_the_settings_that_start_no_job(
    monkeypatch: pytest.MonkeyPatch, login_calls: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(config, "HOPSWORKS_HOST", "example.test")
    store = HopsworksStore()
    store._features_group()

    created = store._store.group_settings
    assert GROUP_SETTINGS.items() <= created.items()
    assert created["stream"] is False
    assert created["statistics_config"] is False
    assert created["online_enabled"] is False
