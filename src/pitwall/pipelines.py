"""Composed pipeline steps shared by the CLI and Dagster assets.

Business logic lives in the layer modules; this module only wires I/O together.
"""

from __future__ import annotations

import json
import logging

from pyspark.sql import functions as F

from pitwall.bronze.writer import append_to_bronze, make_batch_id
from pitwall.config import get_settings
from pitwall.ingestion import fastf1_loader
from pitwall.ingestion.jolpica import JolpicaClient
from pitwall.paths import table_path
from pitwall.quality import QualityReport, apply_expectations
from pitwall.silver import laps as silver_laps
from pitwall.silver.merge import append_quarantine, merge_into_delta
from pitwall.spark import get_spark

log = logging.getLogger(__name__)


def ingest_jolpica(season: int, endpoint: str, round_: int | None = None) -> int:
    settings = get_settings()
    spark = get_spark()
    client = JolpicaClient(settings.jolpica)

    rows = [
        (endpoint, season, round_, request["offset"], json.dumps(payload), request["url"])
        for request, payload in client.iter_pages(endpoint, season, round_)
    ]
    df = spark.createDataFrame(
        rows, "endpoint string, season int, round int, page_offset int, payload string, url string"
    )
    request = {"endpoint": endpoint, "season": season, "round": round_}
    batch_id = make_batch_id("jolpica", request, "".join(r[4] for r in rows))
    written = append_to_bronze(
        df, f"jolpica_{endpoint}", "jolpica", batch_id, request, partition_by=["season"]
    )
    log.info("bronze.jolpica_%s season=%s: %s pages", endpoint, season, written)
    return written


def ingest_fastf1_laps(season: int, round_: int, session_type: str = "R") -> int:
    settings = get_settings()
    spark = get_spark()
    pdf = fastf1_loader.load_session_laps(season, round_, session_type, settings.fastf1.cache_dir)
    request = {"season": season, "round": round_, "session_type": session_type}
    batch_id = make_batch_id("fastf1", request, pdf.to_csv(index=False))
    return append_to_bronze(
        spark.createDataFrame(pdf),
        "fastf1_laps",
        "fastf1",
        batch_id,
        request,
        partition_by=["season"],
    )


def build_silver_laps(season: int) -> QualityReport:
    settings = get_settings()
    spark = get_spark()
    bronze = spark.read.format("delta").load(table_path("bronze", "fastf1_laps"))
    typed = silver_laps.transform_laps(bronze.filter(F.col("season") == season))

    valid, quarantined, report = apply_expectations(
        typed, silver_laps.EXPECTATIONS, settings.quality.max_quarantine_ratio
    )
    merge_into_delta(
        spark, valid, table_path("silver", "laps"), silver_laps.NATURAL_KEY, partition_by=["season"]
    )
    if report.quarantined_rows:
        append_quarantine(quarantined, table_path("quarantine", "laps"))
    log.info("silver.laps season=%s: %s", season, report)
    return report
