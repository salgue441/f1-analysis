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
from pitwall.ingestion.jolpica import ENDPOINTS, JolpicaClient
from pitwall.paths import table_path
from pitwall.quality import QualityReport, apply_expectations
from pitwall.silver.merge import append_quarantine, merge_into_delta
from pitwall.silver.registry import SPECS, target_table
from pitwall.spark import get_spark

log = logging.getLogger(__name__)

# Jolpica endpoints loaded once per season (the rest are per round).
SEASON_ENDPOINTS = [e for e, path in ENDPOINTS.items() if "{round}" not in path]


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
    log.info("bronze.jolpica_%s season=%s round=%s: %s pages", endpoint, season, round_, written)
    return written


def ingest_jolpica_season(season: int, rounds: list[int] | None = None) -> dict[str, int]:
    """All season-level endpoints, plus per-round pit stops for `rounds` (default: all)."""
    pages = {endpoint: ingest_jolpica(season, endpoint) for endpoint in SEASON_ENDPOINTS}
    for round_ in rounds if rounds is not None else season_rounds(season):
        pages["pitstops"] = pages.get("pitstops", 0) + ingest_jolpica(season, "pitstops", round_)
    return pages


def season_rounds(season: int) -> list[int]:
    """Rounds of a season, read from silver.races (so ingest + build races first)."""
    spark = get_spark()
    races = spark.read.format("delta").load(table_path("silver", "races"))
    return [r["round"] for r in races.filter(F.col("season") == season).orderBy("round").collect()]


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


def build_silver(name: str, season: int) -> QualityReport:
    """bronze -> quality gate -> MERGE into silver, for one season of one spec."""
    spec = SPECS[name]
    settings = get_settings()
    spark = get_spark()
    bronze = spark.read.format("delta").load(table_path("bronze", spec.bronze_table))
    typed = spec.transform(bronze.filter(F.col(spec.bronze_season_col) == season))

    valid, quarantined, report = apply_expectations(
        typed, spec.expectations, settings.quality.max_quarantine_ratio
    )
    target = target_table(spec)
    merge_into_delta(
        spark, valid, table_path("silver", target), spec.natural_key, spec.partition_by
    )
    if report.quarantined_rows:
        append_quarantine(quarantined, table_path("quarantine", target))
    log.info("silver.%s (%s) season=%s: %s", target, name, season, report)
    return report


def build_silver_laps(season: int) -> QualityReport:
    return build_silver("laps", season)
