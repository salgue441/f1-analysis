"""Deterministic surrogate keys (SAD §6.4)."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

_NULL_TOKEN = "__null__"


def surrogate_key(*cols: str | Column) -> Column:
    """xxhash64 over the natural key columns; stable across rebuilds.

    Hashes values, not names, so `round` and `event_round` holding the same number give the
    same key.
    """
    exprs = [F.col(c) if isinstance(c, str) else c for c in cols]
    return F.xxhash64(*[F.coalesce(e.cast("string"), F.lit(_NULL_TOKEN)) for e in exprs])
