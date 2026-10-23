"""Persist one row per quality-gate run so pass rates can be tracked over time (SAD §9.1)."""

from __future__ import annotations

import json

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from pitwall.quality.expectations import QualityReport

SCHEMA = (
    "table string, season int, total_rows long, quarantined_rows long, "
    "quarantine_ratio double, failures string"
)


def report_frame(
    spark: SparkSession, table: str, season: int | None, report: QualityReport
) -> DataFrame:
    row = (
        table,
        season,
        report.total_rows,
        report.quarantined_rows,
        report.quarantine_ratio,
        json.dumps(report.failures, sort_keys=True),
    )
    return spark.createDataFrame([row], SCHEMA).withColumn("run_at", F.current_timestamp())


def append_report(
    spark: SparkSession, path: str, table: str, season: int | None, report: QualityReport
) -> None:
    report_frame(spark, table, season, report).write.format("delta").mode("append").save(path)
