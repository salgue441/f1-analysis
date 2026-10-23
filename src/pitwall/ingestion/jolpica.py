"""Rate-limited, paginated client for the Jolpica-F1 (Ergast-compatible) API."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import requests

from pitwall.config import JolpicaSettings

log = logging.getLogger(__name__)

# Endpoint name -> path template. Pit stops are only served per round.
ENDPOINTS: dict[str, str] = {
    "races": "{season}/races",
    "results": "{season}/results",
    "qualifying": "{season}/qualifying",
    "sprint": "{season}/sprint",
    "drivers": "{season}/drivers",
    "constructors": "{season}/constructors",
    "circuits": "{season}/circuits",
    "pitstops": "{season}/{round}/pitstops",
    "laps": "{season}/{round}/laps",
}


class JolpicaClient:
    def __init__(self, settings: JolpicaSettings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        wait = self.settings.min_interval_s - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.settings.max_retries + 1):
            self._throttle()
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                backoff = 2**attempt * 5
                log.warning("Jolpica %s on %s, retrying in %ss", response.status_code, url, backoff)
                time.sleep(backoff)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(
            f"Jolpica request failed after {self.settings.max_retries} retries: {url}"
        )

    def iter_pages(
        self, endpoint: str, season: int, round_: int | None = None
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Yield (request, payload) for every page of an endpoint."""
        if endpoint not in ENDPOINTS:
            raise ValueError(f"Unknown endpoint {endpoint!r}; expected one of {sorted(ENDPOINTS)}")
        template = ENDPOINTS[endpoint]
        if "{round}" in template and round_ is None:
            raise ValueError(f"Endpoint {endpoint!r} requires a round")
        url = f"{self.settings.base_url}/{template.format(season=season, round=round_)}.json"

        offset = 0
        while True:
            params = {"limit": self.settings.page_size, "offset": offset}
            payload = self._get(url, params)
            yield {"url": url, **params}, payload
            total = int(payload["MRData"]["total"])
            offset += self.settings.page_size
            if offset >= total:
                break
