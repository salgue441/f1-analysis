"""Physical-layout evidence for the engineering notes (roadmap Phase 2, SAD §6.3).

Measures, before and after OPTIMIZE + Z-order on the lap tables:
- file count and size (Delta log, no scan),
- files actually read by a typical selective query (`input_file_name`), i.e. data skipping,
- wall time of that query (median of a few runs, after a warm-up),
and the skew of the keys the pipelines group and join on.
"""

from __future__ import annotations

import contextlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pitwall import maintenance
from pitwall.paths import table_path
from pitwall.spark import get_spark

REPORTS = Path(__file__).resolve().parents[2] / "reports"

# One driver's laps in one race: the access pattern of the app and the RQ features.
QUERIES: dict[tuple[str, str], str] = {
    ("silver", "laps"): "season = 2024 AND event_round = 12 AND driver_number = 16",
    ("gold", "fact_lap"): "season = 2024 AND event_round = 12 AND driver_id = 'leclerc'",
    ("silver", "telemetry_car"): "season = 2024 AND event_round = 12 AND driver_number = 16",
    ("gold", "fact_telemetry_sample"): "season = 2024 AND event_round = 12 AND driver_number = 16",
}


def _probe(df: DataFrame, condition: str, repeats: int = 5) -> dict[str, Any]:
    selected = df.filter(condition)
    files = selected.select(F.input_file_name().alias("f")).distinct().count()
    measure = "lap_time_s" if "lap_time_s" in df.columns else "speed_kmh"
    selected.agg(F.sum(measure)).collect()  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        selected.agg(F.sum(measure)).collect()
        times.append(time.perf_counter() - start)
    return {"files_read": files, "query_s_median": round(statistics.median(times), 3)}


def layout_report() -> dict[str, Any]:
    spark = get_spark()
    out: dict[str, Any] = {}
    for (layer, table), condition in QUERIES.items():
        path = table_path(layer, table)  # type: ignore[arg-type]
        if not DeltaTable.isDeltaTable(spark, path):
            continue
        before = {
            **maintenance.table_detail(spark, path),
            **_probe(spark.read.format("delta").load(path), condition),
        }
        maintenance.optimize(spark, path, maintenance.ZORDER.get((layer, table)))
        after = {
            **maintenance.table_detail(spark, path),
            **_probe(spark.read.format("delta").load(path), condition),
        }
        out[f"{layer}.{table}"] = {
            "query": condition,
            "zorder": maintenance.ZORDER.get((layer, table)),
            "before": before,
            "after": after,
        }

    laps = spark.read.format("delta").load(table_path("gold", "fact_lap"))
    out["skew"] = {
        "laps_per_race": maintenance.key_skew(laps, ["season", "event_round"]),
        "laps_per_driver_race": maintenance.key_skew(laps, ["season", "event_round", "driver_id"]),
        "laps_per_circuit": maintenance.key_skew(
            laps.join(
                spark.read.format("delta")
                .load(table_path("gold", "dim_session"))
                .filter(F.col("session_type") == "R")
                .select("season", F.col("round").alias("event_round"), "circuit_id"),
                ["season", "event_round"],
            ),
            ["circuit_id"],
        ),
    }
    telemetry = table_path("gold", "fact_telemetry_sample")
    with contextlib.suppress(Exception):  # telemetry is optional
        out["fact_telemetry_sample"] = maintenance.table_detail(spark, telemetry)
    dest = REPORTS / "engineering"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "layout.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out
