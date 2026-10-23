"""Referential integrity between gold facts and dimensions (SAD §6.1)."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class IntegrityError(RuntimeError):
    pass


def orphan_count(fact: DataFrame, dim: DataFrame, key: str) -> int:
    """Fact rows whose non-null `key` has no match in `dim`."""
    return (
        fact.filter(F.col(key).isNotNull())
        .join(dim.select(key).distinct(), key, "left_anti")
        .count()
    )


def check_foreign_keys(fact: DataFrame, dims: dict[str, DataFrame]) -> dict[str, int]:
    """`dims` maps key column -> dimension. Raises if any fact row is orphaned."""
    orphans = {key: orphan_count(fact, dim, key) for key, dim in dims.items()}
    bad = {k: v for k, v in orphans.items() if v}
    if bad:
        raise IntegrityError(f"Orphaned foreign keys: {bad}")
    return orphans
