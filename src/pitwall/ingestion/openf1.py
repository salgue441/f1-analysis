"""Rate-limited client for the OpenF1 API (historical data from 2023, no account needed).

OpenF1 replaced FastF1 as the lap/tyre/weather/telemetry source (ADR-0006). Every endpoint
is filtered by `session_key`; telemetry endpoints additionally by `driver_number`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from pitwall.config import OpenF1Settings

log = logging.getLogger(__name__)

# OpenF1 session_name -> session code used across the lakehouse (FastF1 naming).
SESSION_CODES = {
    "Race": "R",
    "Sprint": "S",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SS",
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
}
SESSION_NAMES = {code: name for name, code in SESSION_CODES.items()}

# Endpoints loaded once per session, and per driver (telemetry: ~35k samples each).
SESSION_ENDPOINTS = ["drivers", "laps", "stints", "pit", "position", "race_control", "weather"]
TELEMETRY_ENDPOINTS = ["car_data", "location"]


class OpenF1Client:
    def __init__(self, settings: OpenF1Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        wait = self.settings.min_interval_s - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def get(self, endpoint: str, **params: Any) -> tuple[str, list[dict[str, Any]], str]:
        """(url, records, raw body). 404 means 'no data' in OpenF1 and returns no records."""
        url = f"{self.settings.base_url}/{endpoint}"
        for attempt in range(self.settings.max_retries + 1):
            self._throttle()
            response = self.session.get(url, params=params, timeout=60)
            if response.status_code == 429 or response.status_code >= 500:
                backoff = 2**attempt * 5
                log.warning("OpenF1 %s on %s, retrying in %ss", response.status_code, url, backoff)
                time.sleep(backoff)
                continue
            if response.status_code == 404:
                return response.url, [], "[]"
            response.raise_for_status()
            return response.url, response.json(), response.text
        raise RuntimeError(
            f"OpenF1 request failed after {self.settings.max_retries} retries: {url}"
        )

    def sessions(self, year: int, session_type: str) -> list[dict[str, Any]]:
        _, records, _ = self.get("sessions", year=year, session_name=SESSION_NAMES[session_type])
        return [s for s in records if not s.get("is_cancelled")]
