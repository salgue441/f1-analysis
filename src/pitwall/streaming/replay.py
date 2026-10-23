"""Replay a past race from bronze as if it were live.

Each OpenF1 record is released at the moment it would have been published during the race:

- laps: when the lap is completed (`date_start + lap_duration`; a lap without a duration is
  released when the driver's next lap starts);
- position, pit, race_control: their `date`;
- stints: when the stint's first lap starts. Only the stint's start fields are used
  downstream; `lap_end` is the final value and would leak the future;
- drivers: at the start.

The replay clock runs `speed` times faster than real time and flushes everything due into one
landing file every `flush_every_s` seconds of wall time.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pitwall.streaming.landing import LandingWriter


@dataclass(frozen=True)
class Event:
    at: dt.datetime
    endpoint: str
    record: dict[str, Any]


def _ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def release_times(payloads: dict[str, list[dict[str, Any]]]) -> list[Event]:
    """OpenF1 responses per endpoint -> events in release order."""
    events: list[Event] = []
    lap_starts: dict[tuple[int, int], dt.datetime] = {}
    for lap in payloads.get("laps", []):
        start = _ts(lap.get("date_start"))
        if start is not None:
            lap_starts[(lap["driver_number"], lap["lap_number"])] = start

    for lap in payloads.get("laps", []):
        start, duration = _ts(lap.get("date_start")), lap.get("lap_duration")
        if start is not None and duration is not None:
            at = start + dt.timedelta(seconds=duration)
        else:
            at = lap_starts.get((lap["driver_number"], lap["lap_number"] + 1))
        if at is not None:
            events.append(Event(at, "laps", lap))
    for endpoint in ("position", "pit", "race_control"):
        for rec in payloads.get(endpoint, []):
            at = _ts(rec.get("date"))
            if at is not None:
                events.append(Event(at, endpoint, rec))

    start = min((e.at for e in events), default=None)
    if start is None:
        return []
    for stint in payloads.get("stints", []):
        at = lap_starts.get((stint["driver_number"], stint.get("lap_start") or 1), start)
        events.append(Event(at, "stints", stint))
    for driver in payloads.get("drivers", []):
        events.append(Event(start, "drivers", driver))
    # Drivers first at equal times, so every later record has a name to join to.
    order = {"drivers": 0, "stints": 1}
    return sorted(events, key=lambda e: (e.at, order.get(e.endpoint, 2)))


def replay(
    events: list[Event],
    writer: LandingWriter,
    speed: float = 30.0,
    flush_every_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    stop: Callable[[], bool] = lambda: False,
) -> int:
    """Release `events` into landing files on an accelerated clock. Returns files written."""
    if not events:
        return 0
    race_start, wall_start = events[0].at, clock()
    i = 0
    while i < len(events) and not stop():
        sleep(flush_every_s)
        race_now = race_start + dt.timedelta(seconds=(clock() - wall_start) * speed)
        due = []
        while i < len(events) and events[i].at <= race_now:
            due.append((events[i].endpoint, events[i].record))
            i += 1
        writer.write(due)
    return writer.files_written
