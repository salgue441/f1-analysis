"""Backward as-of join in plain Spark (no native ASOF JOIN in Spark 3.5).

Rows of both sides are unioned and sorted by time within each group; each left row then
takes the last non-null right values seen at or before its own time. One window, no range
join, and it is point-in-time correct by construction: a right row later than the left row
can never be picked.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def asof_backward(
    left: DataFrame,
    right: DataFrame,
    by: list[str],
    left_time: str,
    right_time: str,
    values: list[str],
) -> DataFrame:
    """Add `values` from the latest `right` row with right_time <= left_time (same `by`)."""
    left_cols = left.columns
    clashes = set(values) & set(left_cols)
    if clashes:
        raise ValueError(f"Value columns already on the left side: {sorted(clashes)}")

    lhs = left.select(
        *left_cols,
        F.col(left_time).alias("_t"),
        F.lit(1).alias("_side"),
        *[F.lit(None).cast(right.schema[v].dataType).alias(v) for v in values],
    )
    rhs = right.select(
        *[F.col(b) for b in by],
        *[F.lit(None).cast(left.schema[c].dataType).alias(c) for c in left_cols if c not in by],
        F.col(right_time).alias("_t"),
        F.lit(0).alias("_side"),
        *[F.col(v) for v in values],
    )
    # Right rows sort before left rows at equal times, so an exact match is included.
    w = (
        Window.partitionBy(*by)
        .orderBy("_t", "_side")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    rhs = rhs.select(lhs.columns)
    combined = lhs.unionByName(rhs)
    filled = combined.select(
        *[F.col(c) for c in left_cols],
        "_side",
        *[F.last(v, ignorenulls=True).over(w).alias(v) for v in values],
    )
    return filled.filter(F.col("_side") == 1).drop("_side")
