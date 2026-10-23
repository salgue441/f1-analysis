"""Composed pipeline steps shared by the CLI and Dagster assets.

Business logic lives in the layer modules; this module only wires I/O together.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from pitwall import maintenance
from pitwall.bronze.writer import append_to_bronze, make_batch_id
from pitwall.config import get_settings
from pitwall.gold import dims, facts, lap_facts
from pitwall.gold.integrity import check_foreign_keys, check_unique
from pitwall.gold.writer import overwrite_season, overwrite_table
from pitwall.ingestion import fastf1_loader, openf1
from pitwall.ingestion.jolpica import ENDPOINTS, JolpicaClient
from pitwall.ingestion.openf1 import OpenF1Client
from pitwall.paths import Layer, table_path
from pitwall.quality import QualityReport, apply_expectations
from pitwall.quality.log import append_report
from pitwall.silver import driver_id_map
from pitwall.silver.merge import append_quarantine, merge_into_delta, replace_partitions
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


FASTF1_TELEMETRY_TABLES = set(fastf1_loader.TELEMETRY_FRAMES)


def ingest_fastf1_session(
    season: int, round_: int, session_type: str = "R", telemetry: bool = False
) -> dict[str, int]:
    """One FastF1 session load -> every bronze table it feeds (laps, weather, status...)."""
    settings = get_settings()
    spark = get_spark()
    frames = fastf1_loader.load_session(
        season, round_, session_type, settings.fastf1.cache_dir, telemetry=telemetry
    )
    request = {"season": season, "round": round_, "session_type": session_type}
    written = {}
    for table, pdf in frames.items():
        if pdf.empty:
            log.warning("bronze.%s %s: empty frame, skipped", table, request)
            continue
        batch_id = make_batch_id("fastf1", {**request, "table": table}, pdf.to_csv(index=False))
        partitions = (
            ["season", "event_round", "session_type"]
            if table in FASTF1_TELEMETRY_TABLES
            else ["season"]
        )
        written[table] = append_to_bronze(
            spark.createDataFrame(pdf), table, "fastf1", batch_id, request, partitions
        )
    log.info("bronze fastf1 %s: %s", request, written)
    return written


def ingest_fastf1_laps(season: int, round_: int, session_type: str = "R") -> int:
    return ingest_fastf1_session(season, round_, session_type).get("fastf1_laps", 0)


def build_silver(name: str, season: int) -> QualityReport:
    """bronze -> quality gate -> MERGE into silver, for one season of one spec."""
    spec = SPECS[name]
    settings = get_settings()
    spark = get_spark()
    tables = (spec.bronze_table,) if isinstance(spec.bronze_table, str) else spec.bronze_table
    in_season = F.col(spec.bronze_season_col) == season
    typed = spec.transform(*[_read("bronze", t).filter(in_season) for t in tables])

    valid, quarantined, report = apply_expectations(
        typed, spec.expectations, settings.quality.max_quarantine_ratio
    )
    target = target_table(spec)
    if spec.replace_partitions and spec.partition_by:
        replace_partitions(valid, table_path("silver", target), spec.partition_by)
    else:
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


# --- OpenF1 ----------------------------------------------------------------------------------

OPENF1_SCHEMA = (
    "season int, event_round int, session_type string, session_key int, "
    "session_start_utc timestamp, driver_number int, payload string, url string"
)


def match_rounds(
    sessions: list[dict], races: list[dict], session_type: str, max_gap_hours: float = 36
) -> dict[int, int]:
    """OpenF1 session_key -> Jolpica round, by nearest scheduled start in the same season.

    OpenF1 has no round numbers, and names differ between sources; start times agree.
    """
    start_col = dims.SESSION_START_COLUMNS[session_type]
    matched = {}
    for s in sessions:
        start = dt.datetime.fromisoformat(s["date_start"]).replace(tzinfo=None)
        candidates = [
            (abs((r[start_col] - start).total_seconds()), r["round"])
            for r in races
            if r.get(start_col) is not None
        ]
        if not candidates:
            continue
        gap_s, round_ = min(candidates)
        if gap_s <= max_gap_hours * 3600:
            matched[s["session_key"]] = round_
        else:
            log.warning("OpenF1 session %s has no Jolpica round within %sh", s, max_gap_hours)
    return matched


def ingest_openf1_season(
    season: int,
    rounds: list[int] | None = None,
    telemetry: bool = False,
    endpoints: list[str] | None = None,
) -> dict[str, int]:
    """Every configured session type of a season -> bronze.openf1_<endpoint>.

    Needs silver.races for the season (to number the sessions). Each (session, endpoint) is
    one bronze batch; telemetry fetches every driver and writes them as one batch.
    """
    settings = get_settings()
    spark = get_spark()
    client = OpenF1Client(settings.openf1)
    races = [
        r.asDict() for r in _read("silver", "races").filter(F.col("season") == season).collect()
    ]
    if endpoints is None:
        endpoints = openf1.SESSION_ENDPOINTS + (openf1.TELEMETRY_ENDPOINTS if telemetry else [])
    written: dict[str, int] = {}
    for session_type in settings.openf1.session_types:
        sessions = {s["session_key"]: s for s in client.sessions(season, session_type)}
        matched = match_rounds(list(sessions.values()), races, session_type)
        for session_key, round_ in sorted(matched.items(), key=lambda kv: kv[1]):
            if rounds is not None and round_ not in rounds:
                continue
            start = dt.datetime.fromisoformat(sessions[session_key]["date_start"])
            start = start.replace(tzinfo=None)
            drivers: list[int] | None = None
            for endpoint in endpoints:
                calls: list[tuple[int | None, dict]] = [(None, {})]
                if endpoint in openf1.TELEMETRY_ENDPOINTS:
                    if drivers is None:
                        _, recs, _ = client.get("drivers", session_key=session_key)
                        drivers = sorted({d["driver_number"] for d in recs})
                    calls = [(n, {"driver_number": n}) for n in drivers]
                rows = []
                for driver_number, extra in calls:
                    url, _, body = client.get(endpoint, session_key=session_key, **extra)
                    rows.append(
                        (season, round_, session_type, session_key, start, driver_number, body, url)
                    )
                request = {"session_key": session_key, "endpoint": endpoint}
                batch_id = make_batch_id("openf1", request, "".join(r[6] for r in rows))
                partitions = (
                    ["season", "event_round", "session_type"]
                    if endpoint in openf1.TELEMETRY_ENDPOINTS
                    else ["season"]
                )
                written[endpoint] = written.get(endpoint, 0) + append_to_bronze(
                    spark.createDataFrame(rows, OPENF1_SCHEMA),
                    f"openf1_{endpoint}",
                    "openf1",
                    batch_id,
                    request,
                    partitions,
                )
            log.info("bronze openf1 %s R%s %s: loaded", season, round_, session_type)
    return written


# --- gold ------------------------------------------------------------------------------------


def _read(layer: Layer, table: str) -> DataFrame:
    return get_spark().read.format("delta").load(table_path(layer, table))


def _in_season(season: int) -> Column:
    """Season filter; starts Spark first, since a Column needs an active session."""
    get_spark()
    return F.col("season") == season


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
    in_season = _in_season(season)
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


def build_driver_id_map(season: int) -> int:
    in_season = _in_season(season)
    mapping = driver_id_map.build_driver_id_map(
        _read("silver", "results").filter(in_season),
        _read("silver", "qualifying").filter(in_season),
        _read("silver", "drivers"),
    )
    driver_id_map.check_one_to_one(mapping)
    path = table_path("silver", "driver_id_map")
    overwrite_season(get_spark(), mapping, path, season)
    return _read("silver", "driver_id_map").filter(in_season).count()


FACT_KEYS = {
    "fact_lap": "lap_key",
    "fact_stint": "stint_key",
    "fact_pit_stop": "pit_stop_key",
}


def _write_fact(name: str, fact: DataFrame, season: int, fks: dict[str, str]) -> int:
    """PK + FK checks against the named gold dims, then replace the season partition."""
    check_unique(fact, FACT_KEYS[name])
    check_foreign_keys(fact, {key: _read("gold", dim) for key, dim in fks.items()})
    overwrite_season(get_spark(), fact, table_path("gold", name), season)
    count = _read("gold", name).filter(F.col("season") == season).count()
    log.info("gold.%s season=%s: %s rows", name, season, count)
    return count


def build_fact_lap(season: int) -> int:
    in_season = _in_season(season)
    fact = lap_facts.build_fact_lap(
        _read("silver", "laps").filter(in_season),
        _read("silver", "weather").filter(in_season),
        _read("silver", "driver_id_map").filter(in_season),
    )
    return _write_fact(
        "fact_lap",
        fact,
        season,
        {
            "session_key": "dim_session",
            "driver_key": "dim_driver",
            "constructor_key": "dim_constructor",
            "compound_key": "dim_compound",
        },
    )


def build_fact_stint(season: int) -> int:
    fact = lap_facts.build_fact_stint(_read("gold", "fact_lap").filter(F.col("season") == season))
    return _write_fact(
        "fact_stint",
        fact,
        season,
        {"session_key": "dim_session", "driver_key": "dim_driver", "compound_key": "dim_compound"},
    )


def build_fact_pit_stop(season: int) -> int:
    in_season = _in_season(season)
    fact = lap_facts.build_fact_pit_stop(
        _read("silver", "pit_stops").filter(in_season),
        _read("gold", "fact_lap").filter(in_season),
        _read("silver", "driver_id_map").filter(in_season),
    )
    return _write_fact(
        "fact_pit_stop", fact, season, {"session_key": "dim_session", "driver_key": "dim_driver"}
    )


def build_fact_telemetry_sample(season: int) -> int:
    """Telemetry is partitioned by session, so it replaces partitions instead of a season."""
    in_season = _in_season(season)
    fact = lap_facts.build_fact_telemetry_sample(
        _read("silver", "telemetry_car").filter(in_season),
        _read("silver", "telemetry_pos").filter(in_season),
    )
    path = table_path("gold", "fact_telemetry_sample")
    replace_partitions(fact, path, ["season", "event_round", "session_type"])
    return _read("gold", "fact_telemetry_sample").filter(in_season).count()


def ingest_fastf1_season(
    season: int, session_types: list[str] | None = None, telemetry: bool = False
) -> int:
    """Every round of a season; a session FastF1 can't load is logged and skipped."""
    sessions = session_types or get_settings().fastf1.session_types
    loaded = 0
    for round_ in fastf1_loader.event_rounds(season):
        for session_type in sessions:
            try:
                ingest_fastf1_session(season, round_, session_type, telemetry)
                loaded += 1
            except Exception as exc:  # noqa: BLE001 - one bad session must not stop a backfill
                log.error("fastf1 %s R%s %s failed: %s", season, round_, session_type, exc)
    return loaded


def run_season_laps(season: int, ingest: bool = True, telemetry: bool = False) -> None:
    """The Phase 2 graph for one season (needs Phase 1 for that season first)."""
    if ingest:
        ingest_openf1_season(season, telemetry=telemetry)
    for name in ["laps", "weather", "race_control"]:
        build_silver(name, season)
    build_driver_id_map(season)
    build_fact_lap(season)
    build_fact_stint(season)
    build_fact_pit_stop(season)
    if telemetry:
        build_silver("telemetry_car", season)
        build_silver("telemetry_pos", season)
        build_fact_telemetry_sample(season)


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
    spark = get_spark()
    return {
        f"{layer}.{table}": table_fingerprint(spark.read.format("delta").load(path))
        for layer, table, path in _delta_tables(layers)
    }


def _delta_tables(layers: tuple[Layer, ...]) -> list[tuple[str, str, str]]:
    """(layer, table, path) for every Delta table under the lakehouse root."""
    from delta.tables import DeltaTable

    spark = get_spark()
    root = get_settings().lakehouse_root
    hadoop_path = spark._jvm.org.apache.hadoop.fs.Path  # type: ignore[union-attr]
    fs = hadoop_path(root).getFileSystem(spark._jsc.hadoopConfiguration())  # type: ignore[union-attr]
    found = []
    for layer in layers:
        layer_dir = hadoop_path(f"{root}/{layer}")
        if not fs.exists(layer_dir):
            continue
        for status in fs.listStatus(layer_dir):
            path = status.getPath().toString()
            if DeltaTable.isDeltaTable(spark, path):
                found.append((layer, status.getPath().getName(), path))
    return sorted(found)


def maintain(vacuum_tables: bool = True) -> dict[str, dict]:
    """OPTIMIZE (Z-order where configured) and VACUUM every table; returns before/after."""
    spark = get_spark()
    out = {}
    for layer, table, path in _delta_tables(("bronze", "silver", "gold", "features")):
        out[f"{layer}.{table}"] = maintenance.optimize(
            spark, path, maintenance.ZORDER.get((layer, table))
        )
        if vacuum_tables:
            maintenance.vacuum(spark, path)
    return out
