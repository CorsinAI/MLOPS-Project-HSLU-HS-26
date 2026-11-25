"""Login and group settings. The Hopsworks client itself is replaced by a fake."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import requests
from hopsworks_common.client.exceptions import RestAPIError

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


class FakeResponse:
    reason = "Server said no"
    content = b"{}"

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return {}


class FlakyGroup:
    """Fails with each given HTTP status in turn, then succeeds."""

    def __init__(self, *statuses: int) -> None:
        self.statuses = list(statuses)
        self.inserts = 0

    def insert(self, rows: Any) -> None:
        self.inserts += 1
        if self.statuses:
            response = FakeResponse(self.statuses.pop(0))
            raise RestAPIError("https://example.test", response)


@pytest.fixture
def no_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hopsworks_store, "WRITE_RETRY_DELAY_S", 0)


@pytest.mark.usefixtures("no_delay")
def test_a_server_error_is_retried() -> None:
    group = FlakyGroup(500, 503)
    hopsworks_store._insert(group, pd.DataFrame())
    assert group.inserts == 3


@pytest.mark.usefixtures("no_delay")
def test_a_lasting_server_error_fails_the_run() -> None:
    group = FlakyGroup(500, 500, 500, 500)
    with pytest.raises(RestAPIError):
        hopsworks_store._insert(group, pd.DataFrame())
    assert group.inserts == hopsworks_store.WRITE_ATTEMPTS


@pytest.mark.usefixtures("no_delay")
def test_a_client_error_is_not_retried() -> None:
    group = FlakyGroup(403)
    with pytest.raises(RestAPIError):
        hopsworks_store._insert(group, pd.DataFrame())
    assert group.inserts == 1


class RaisingGroup:
    """Raises each given exception in turn, then succeeds."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.inserts = 0

    def insert(self, rows: Any) -> None:
        self.inserts += 1
        if self.errors:
            raise self.errors.pop(0)


@pytest.mark.usefixtures("no_delay")
@pytest.mark.parametrize(
    "error",
    [
        # Run 42458850154: the connection dropped during the label write.
        requests.exceptions.ConnectionError("Connection aborted."),
        requests.exceptions.Timeout("Read timed out."),
        # Run 44103070337: delta-rs failed to read the table during the write.
        OSError("Kernel error -> Generic HdfsObjectStore error"),
    ],
)
def test_a_dropped_connection_or_storage_error_is_retried(error: Exception) -> None:
    group = RaisingGroup(error)
    hopsworks_store._insert(group, pd.DataFrame())
    assert group.inserts == 2


@pytest.mark.usefixtures("no_delay")
def test_a_lasting_connection_error_fails_the_run() -> None:
    errors = [requests.exceptions.ConnectionError("down")] * 4
    group = RaisingGroup(*errors)
    with pytest.raises(requests.exceptions.ConnectionError):
        hopsworks_store._insert(group, pd.DataFrame())
    assert group.inserts == hopsworks_store.WRITE_ATTEMPTS


@pytest.mark.usefixtures("no_delay")
def test_a_programming_error_is_not_retried() -> None:
    group = RaisingGroup(ValueError("bad frame"))
    with pytest.raises(ValueError):
        hopsworks_store._insert(group, pd.DataFrame())
    assert group.inserts == 1


@pytest.mark.usefixtures("no_delay")
def test_a_server_error_at_login_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    # Run 45430760114: login got HTTP 503 from the version endpoint.
    monkeypatch.setattr(config, "HOPSWORKS_HOST", "example.test")
    attempts: list[int] = []

    def flaky_login(**settings: Any) -> FakeProject:
        attempts.append(1)
        if len(attempts) == 1:
            raise RestAPIError("https://example.test", FakeResponse(503))
        return FakeProject()

    monkeypatch.setattr(hopsworks_store.hopsworks, "login", flaky_login)
    HopsworksStore()
    assert len(attempts) == 2


@pytest.mark.usefixtures("no_delay")
def test_a_rejected_api_key_at_login_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "HOPSWORKS_HOST", "example.test")
    attempts: list[int] = []

    def rejecting_login(**settings: Any) -> FakeProject:
        attempts.append(1)
        raise RestAPIError("https://example.test", FakeResponse(401))

    monkeypatch.setattr(hopsworks_store.hopsworks, "login", rejecting_login)
    with pytest.raises(RestAPIError):
        HopsworksStore()
    assert len(attempts) == 1
