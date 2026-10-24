"""OpenF1 live feed -> landing files (needs an OpenF1 Sponsor account, ADR-0007).

OpenF1 publishes live data over MQTT (`mqtt.openf1.org:8883`, TLS). The OAuth2 access token
from `POST https://api.openf1.org/token` is the MQTT password, and it expires after an hour,
so the client re-authenticates before then. Topics mirror the REST endpoints (`v1/laps`).
Messages are buffered and flushed to the landing directory every `flush_every_s`, in exactly
the format the replay writes, so the streaming query can't tell them apart.

`paho-mqtt` is an optional dependency (`pip install -e .[live]`).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

from pitwall.streaming.landing import STREAM_ENDPOINTS, LandingWriter

log = logging.getLogger(__name__)

TOKEN_URL = "https://api.openf1.org/token"
BROKER = "mqtt.openf1.org"
PORT = 8883
REFRESH_BEFORE_EXPIRY_S = 300


def fetch_token(username: str, password: str, session: Any = requests) -> tuple[str, float]:
    """(access token, seconds until it expires)."""
    response = session.post(
        TOKEN_URL,
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    body = response.json()
    return body["access_token"], float(body.get("expires_in", 3600))


class LiveBuffer:
    """Thread-safe buffer of (endpoint, record) from MQTT callbacks, flushed to landing."""

    def __init__(self, writer: LandingWriter, session_key: int | None = None):
        self.writer = writer
        self.session_key = session_key
        self._lock = threading.Lock()
        self._pending: list[tuple[str, dict[str, Any]]] = []

    def on_message(self, topic: str, payload: bytes) -> None:
        endpoint = topic.removeprefix("v1/")
        if endpoint not in STREAM_ENDPOINTS:
            return
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("Skipping malformed message on %s", topic)
            return
        if self.session_key is not None and record.get("session_key") != self.session_key:
            return
        with self._lock:
            self._pending.append((endpoint, record))

    def flush(self) -> int:
        with self._lock:
            pending, self._pending = self._pending, []
        self.writer.write(pending)
        return len(pending)


def run_live(
    username: str,
    password: str,
    writer: LandingWriter,
    session_key: int | None = None,
    flush_every_s: float = 1.0,
    stop: Callable[[], bool] = lambda: False,
) -> None:
    """Subscribe to the live feed until `stop()` returns True."""
    import ssl

    import paho.mqtt.client as mqtt

    buffer = LiveBuffer(writer, session_key)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.on_message = lambda _c, _u, msg: buffer.on_message(msg.topic, msg.payload)
    client.on_connect = lambda c, *_: [c.subscribe(f"v1/{e}") for e in STREAM_ENDPOINTS]

    token, expires_in = fetch_token(username, password)
    client.username_pw_set(username=username, password=token)
    client.connect(BROKER, PORT)
    client.loop_start()
    refresh_at = time.monotonic() + expires_in - REFRESH_BEFORE_EXPIRY_S
    try:
        while not stop():
            time.sleep(flush_every_s)
            buffer.flush()
            if time.monotonic() >= refresh_at:
                token, expires_in = fetch_token(username, password)
                client.username_pw_set(username=username, password=token)
                client.reconnect()
                refresh_at = time.monotonic() + expires_in - REFRESH_BEFORE_EXPIRY_S
    finally:
        buffer.flush()
        client.loop_stop()
        client.disconnect()
