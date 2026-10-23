"""Idempotent upsert into silver Delta tables (SAD §9.2)."""

from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession


def merge_into_delta(
    spark: SparkSession,
    df: DataFrame,
    path: str,
    keys: list[str],
    partition_by: list[str] | None = None,
) -> None:
    """MERGE `df` into the Delta table at `path` on `keys`; create the table if missing."""
    if not DeltaTable.isDeltaTable(spark, path):
        writer = df.write.format("delta").mode("overwrite")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(path)
        return

    condition = " AND ".join(f"t.{k} <=> s.{k}" for k in keys)
    # Latest ingestion wins regardless of run order: rebuilding an older season must not
    # overwrite an entity row that a later batch (e.g. another season's) already updated.
    newer = "s._ingested_at >= t._ingested_at" if "_ingested_at" in df.columns else None
    (
        DeltaTable.forPath(spark, path)
        .alias("t")
        .merge(df.alias("s"), condition)
        .whenMatchedUpdateAll(condition=newer)
        .whenNotMatchedInsertAll()
        .execute()
    )


def append_quarantine(df: DataFrame, path: str) -> None:
    df.write.format("delta").mode("append").option("mergeSchema", "true").save(path)


def replace_partitions(df: DataFrame, path: str, partition_by: list[str]) -> None:
    """Overwrite only the partitions present in `df` (dynamic partition overwrite).

    Repartitioning by the partition columns first writes one file per partition instead of
    one per shuffle task: 2024 telemetry had ~19 files of ~0.6 MB per race before this.
    """
    (
        df.repartition(*partition_by)
        .write.format("delta")
        .mode("overwrite")
        .option("partitionOverwriteMode", "dynamic")
        .partitionBy(*partition_by)
        .save(path)
    )
