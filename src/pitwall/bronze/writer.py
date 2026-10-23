"""Generic append-only bronze writer with ingestion metadata (SAD §6.1, §9.2)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pitwall.paths import table_path


def make_batch_id(source: str, request: dict[str, Any], content: str | bytes) -> str:
    """Deterministic batch id: the same request + same content always gives the same id."""
    if isinstance(content, str):
        content = content.encode()
    digest = hashlib.sha256()
    digest.update(source.encode())
    digest.update(json.dumps(request, sort_keys=True, default=str).encode())
    digest.update(content)
    return digest.hexdigest()[:16]


def with_ingestion_metadata(
    df: DataFrame, source: str, batch_id: str, request: dict[str, Any]
) -> DataFrame:
    return (
        df.withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source", F.lit(source))
        .withColumn("_batch_id", F.lit(batch_id))
        .withColumn("_request", F.lit(json.dumps(request, sort_keys=True, default=str)))
    )


def append_to_bronze(
    df: DataFrame,
    table: str,
    source: str,
    batch_id: str,
    request: dict[str, Any],
    partition_by: list[str] | None = None,
) -> int:
    """Append a batch to a bronze Delta table. Returns the number of rows written."""
    out = with_ingestion_metadata(df, source, batch_id, request)
    writer = out.write.format("delta").mode("append").option("mergeSchema", "true")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(table_path("bronze", table))
    return out.count()
