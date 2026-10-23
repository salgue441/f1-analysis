"""Delta table maintenance and physical-layout diagnostics (SAD §6.3).

OPTIMIZE compacts small files (FastF1 lands one small batch per session) and Z-orders the
columns queries filter on inside a partition. VACUUM drops files older than the retention.
"""

from __future__ import annotations

from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# (layer, table) -> Z-order columns. Tables not listed are only compacted.
ZORDER: dict[tuple[str, str], list[str]] = {
    ("gold", "fact_lap"): ["event_round", "driver_key"],
    ("gold", "fact_stint"): ["event_round", "driver_key"],
    ("silver", "laps"): ["event_round", "driver_number"],
    ("silver", "telemetry_car"): ["driver_number"],
    ("silver", "telemetry_pos"): ["driver_number"],
    ("gold", "fact_telemetry_sample"): ["driver_number"],
}
MIN_RETENTION_HOURS = 7 * 24


def table_detail(spark: SparkSession, path: str) -> dict[str, Any]:
    """File count, total size and mean file size from the Delta log (no data scan)."""
    d = DeltaTable.forPath(spark, path).detail().first()
    files, size = d["numFiles"] or 0, d["sizeInBytes"] or 0
    return {
        "files": files,
        "size_mb": round(size / 2**20, 2),
        "mean_file_mb": round(size / files / 2**20, 3) if files else 0.0,
        "partition_columns": list(d["partitionColumns"]),
    }


def optimize(spark: SparkSession, path: str, zorder: list[str] | None = None) -> dict[str, Any]:
    before = table_detail(spark, path)
    builder = DeltaTable.forPath(spark, path).optimize()
    if zorder:
        builder.executeZOrderBy(*zorder)
    else:
        builder.executeCompaction()
    return {"before": before, "after": table_detail(spark, path), "zorder": zorder or []}


def vacuum(spark: SparkSession, path: str, retention_hours: int = MIN_RETENTION_HOURS) -> None:
    if retention_hours < MIN_RETENTION_HOURS:
        raise ValueError(f"Retention below {MIN_RETENTION_HOURS}h breaks time travel (SAD §6.3)")
    DeltaTable.forPath(spark, path).vacuum(retention_hours)


def key_skew(df: DataFrame, keys: list[str]) -> dict[str, float]:
    """How unevenly rows spread over `keys`: the max/median ratio is what hurts a shuffle."""
    counts = df.groupBy(*keys).count()
    s = counts.agg(
        F.count(F.lit(1)).alias("groups"),
        F.min("count").alias("min"),
        F.percentile_approx("count", 0.5).alias("median"),
        F.max("count").alias("max"),
    ).first()
    median = s["median"] or 0
    return {
        "groups": s["groups"],
        "min_rows": s["min"],
        "median_rows": median,
        "max_rows": s["max"],
        "max_over_median": round(s["max"] / median, 2) if median else 0.0,
    }
