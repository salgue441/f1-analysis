"""Deterministic surrogate keys (SAD §6.4)."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

_NULL_TOKEN = "__null__"


def surrogate_key(*cols: str) -> Column:
    """xxhash64 over the natural key columns; stable across rebuilds."""
    return F.xxhash64(*[F.coalesce(F.col(c).cast("string"), F.lit(_NULL_TOKEN)) for c in cols])
