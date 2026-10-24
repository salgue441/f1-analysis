"""Table path conventions: <lakehouse_root>/<layer>/<table>."""

from __future__ import annotations

from typing import Literal

from pitwall.config import get_settings

Layer = Literal["bronze", "silver", "gold", "features", "quarantine", "stream"]


def table_path(layer: Layer, table: str, lakehouse_root: str | None = None) -> str:
    root = (lakehouse_root or get_settings().lakehouse_root).rstrip("/")
    return f"{root}/{layer}/{table}"
