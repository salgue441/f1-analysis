"""Composed pipeline steps shared by the CLI and Dagster assets.

Business logic lives in the layer modules; this module only wires I/O together.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pitwall.bronze.writer import append_to_bronze, make_batch_id
from pitwall.config import get_settings
from pitwall.gold import dims, facts
from pitwall.gold.integrity import check_foreign_keys
from pitwall.gold.writer import overwrite_season, overwrite_table
from pitwall.ingestion import fastf1_loader
from pitwall.ingestion.jolpica import ENDPOINTS, JolpicaClient
from pitwall.paths import Layer, table_path
from pitwall.quality import QualityReport, apply_expectations
from pitwall.quality.log import append_report
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
    append_report(spark, table_path("quarantine", "_dq_runs"), name, season, report)
    log.info("silver.%s (%s) season=%s: %s", target, name, season, report)
    return report


def build_silver_laps(season: int) -> QualityReport:
    return build_silver("laps", season)


# --- gold ------------------------------------------------------------------------------------


def _read(layer: Layer, table: str) -> DataFrame:
    return get_spark().read.format("delta").load(table_path(layer, table))


def build_gold_dims() -> dict[str, int]:
    """Rebuild all dimensions from silver (they are small and span seasons)."""
    spark = get_spark()
    seasons = get_settings().seasons
    races, results, qualifying = (
        _read("silver", "races"),
        _read("silver", "results"),
        _read("silver", "qualifying"),
    )
    observed = results.select("season", "round", "session_type").unionByName(
        qualifying.select("season", "round", F.lit("Q").alias("session_type"))
    )
    tables = {
        "dim_driver": dims.build_dim_driver(_read("silver", "drivers")),
        "dim_constructor": dims.build_dim_constructor(_read("silver", "constructors")),
        "dim_circuit": dims.build_dim_circuit(_read("silver", "circuits")),
        "dim_compound": dims.build_dim_compound(spark),
        "dim_date": dims.build_dim_date(
            spark, dt.date(seasons.start, 1, 1), dt.date(seasons.end, 12, 31)
        ),
        "dim_session": dims.build_dim_session(races, results, observed),
        "dim_driver_team": dims.build_dim_driver_team(dims.driver_team_entries(results, races)),
    }
    counts = {}
    for name, df in tables.items():
        overwrite_table(df, table_path("gold", name))
        counts[name] = _read("gold", name).count()
    log.info("gold dims: %s", counts)
    return counts


def build_fact_session_result(season: int) -> int:
    in_season = F.col("season") == season
    fact = facts.build_fact_session_result(
        _read("silver", "results").filter(in_season),
        _read("silver", "qualifying").filter(in_season),
    )
    check_foreign_keys(
        fact,
        {
            "session_key": _read("gold", "dim_session"),
            "driver_key": _read("gold", "dim_driver"),
            "constructor_key": _read("gold", "dim_constructor"),
        },
    )
    overwrite_season(get_spark(), fact, table_path("gold", "fact_session_result"), season)
    return _read("gold", "fact_session_result").filter(in_season).count()


# --- end-to-end runs -------------------------------------------------------------------------

SILVER_FROM_JOLPICA = [
    "races",
    "circuits",
    "drivers",
    "constructors",
    "results",
    "sprint_results",
    "qualifying",
]


def run_season(season: int, ingest: bool = True) -> None:
    """The Phase 1 graph for one season without Dagster (same steps, same order)."""
    if ingest:
        for endpoint in SEASON_ENDPOINTS:
            ingest_jolpica(season, endpoint)
    for name in SILVER_FROM_JOLPICA:
        build_silver(name, season)
    if ingest:
        for round_ in season_rounds(season):
            ingest_jolpica(season, "pitstops", round_)
    build_silver("pit_stops", season)
    build_gold_dims()
    build_fact_session_result(season)


# Columns that legitimately differ between identical rebuilds.
VOLATILE_COLUMNS = {"_ingested_at", "_updated_at", "_quarantined_at", "run_at"}


def table_fingerprint(df: DataFrame) -> dict[str, int]:
    """Row count + order-independent checksum of the non-volatile columns."""
    cols = sorted(c for c in df.columns if c not in VOLATILE_COLUMNS)
    row = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.xxhash64(*cols).cast("decimal(38,0)")).cast("string").alias("checksum"),
    ).first()
    return {"rows": row["rows"], "checksum": row["checksum"]}


def fingerprints(layers: tuple[Layer, ...] = ("silver", "gold")) -> dict[str, dict[str, int]]:
    """Fingerprint every Delta table in the given layers (for rebuild-twice checks)."""
    from delta.tables import DeltaTable

    spark = get_spark()
    root = get_settings().lakehouse_root
    jvm_path = spark._jvm.org.apache.hadoop.fs.Path(root)  # type: ignore[union-attr]
    fs = jvm_path.getFileSystem(spark._jsc.hadoopConfiguration())  # type: ignore[union-attr]
    out = {}
    for layer in layers:
        layer_path = spark._jvm.org.apache.hadoop.fs.Path(f"{root}/{layer}")  # type: ignore[union-attr]
        if not fs.exists(layer_path):
            continue
        for status in fs.listStatus(layer_path):
            path = status.getPath().toString()
            if DeltaTable.isDeltaTable(spark, path):
                name = f"{layer}.{status.getPath().getName()}"
                out[name] = table_fingerprint(spark.read.format("delta").load(path))
    return out
