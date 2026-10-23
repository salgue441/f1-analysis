"""Gold/feature writes: whole-table overwrite for dims, per-season replaceWhere for facts."""

from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def overwrite_table(df: DataFrame, path: str, partition_by: list[str] | None = None) -> None:
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)


def overwrite_season(spark: SparkSession, df: DataFrame, path: str, season: int) -> None:
    """Replace exactly one season partition (SAD §9.2). Rows for other seasons are refused."""
    df = df.filter(F.col("season") == season)
    if not DeltaTable.isDeltaTable(spark, path):
        overwrite_table(df, path, ["season"])
        return
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"season = {int(season)}")
        # New columns are allowed (additive evolution); a changed type still fails.
        .option("mergeSchema", "true")
        .save(path)
    )
