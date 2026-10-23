"""Helpers shared by silver transforms."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F


def latest_per_key(df: DataFrame, keys: list[str], order_col: str = "_ingested_at") -> DataFrame:
    """Keep one row per natural key: the most recently ingested one (SAD §9.2)."""
    latest_first = Window.partitionBy(*keys).orderBy(F.col(order_col).desc())
    return (
        df.withColumn("_rn", F.row_number().over(latest_first))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def duration_s(col: Column) -> Column:
    """'1:23.456', '83.456' or '1:02:03.4' -> seconds. Blank or malformed -> NULL.

    Relies on Spark's non-ANSI cast, where a malformed string casts to NULL.
    """
    parts = F.split(F.trim(col), ":")
    seconds = F.aggregate(
        parts, F.lit(0.0).cast("double"), lambda acc, part: acc * 60 + part.cast("double")
    )
    return F.when(F.trim(col) == "", F.lit(None).cast("double")).otherwise(seconds)


def utc_timestamp(date_col: Column, time_col: Column) -> Column:
    """Jolpica date ('2024-03-02') + optional time ('15:00:00Z') -> UTC timestamp."""
    time_part = F.coalesce(F.regexp_replace(time_col, "Z$", ""), F.lit("00:00:00"))
    # concat (not concat_ws) so a missing date gives NULL instead of a time-only string.
    return F.to_timestamp(F.concat(date_col, F.lit(" "), time_part))
