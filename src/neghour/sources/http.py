"""Shared HTTP helper for source modules: GET JSON with retries."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ATTEMPTS = 3
RETRY_DELAY_S = 2.0
TIMEOUT_S = 60.0
USER_AGENT = "neghour-feature-pipeline"


class SourceUnavailableError(RuntimeError):
    """Raised when a data source cannot be reached after all retries."""


def new_client() -> httpx.Client:
    """Return an HTTP client with the project defaults."""
    return httpx.Client(
        timeout=TIMEOUT_S,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


def get_json(
    client: httpx.Client, url: str, params: dict[str, Any] | None = None
) -> Any:
    """Return the JSON body of a GET request, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
            logger.warning("GET failed, attempt %d of %d: %s", attempt, ATTEMPTS, url)
            if attempt < ATTEMPTS:
                time.sleep(RETRY_DELAY_S * attempt)
    raise SourceUnavailableError(f"GET failed for {url}") from last_error
