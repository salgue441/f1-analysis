"""Landing files: the hand-off between producers and the streaming query.

One file per flush, JSON lines, each line an OpenF1 record plus `_endpoint` and
`_emitted_at`. Files are written under a name Spark ignores (leading `_`) and renamed when
complete, so the file source never reads a half-written file.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Endpoints the race state is built from; the live feed subscribes to the same ones.
STREAM_ENDPOINTS = ["drivers", "laps", "stints", "pit", "position", "race_control"]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class LandingWriter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.files_written = 0
        self.records_written = 0

    def write(
        self, records: Iterable[tuple[str, dict[str, Any]]], emitted_at: dt.datetime | None = None
    ) -> Path | None:
        """Write (endpoint, record) pairs as one landing file. Returns None if empty."""
        emitted = (emitted_at or utc_now()).isoformat()
        lines = [
            json.dumps({**record, "_endpoint": endpoint, "_emitted_at": emitted}, default=str)
            for endpoint, record in records
        ]
        if not lines:
            return None
        name = f"{self.files_written:08d}-{uuid.uuid4().hex[:8]}.json"
        tmp = self.directory / f"_{name}.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        final = self.directory / name
        os.replace(tmp, final)
        self.files_written += 1
        self.records_written += len(lines)
        return final
