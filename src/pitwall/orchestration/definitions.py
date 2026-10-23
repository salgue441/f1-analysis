"""Dagster asset graph (ADR-0004). Assets are thin wrappers over `pitwall.pipelines`.

No `from __future__ import annotations` here: Dagster inspects the context type hint at runtime.
"""

from dagster import (
    AssetExecutionContext,
    Definitions,
    MaterializeResult,
    StaticPartitionsDefinition,
    asset,
)

from pitwall import pipelines
from pitwall.config import get_settings
from pitwall.ingestion.fastf1_loader import event_rounds

seasons = StaticPartitionsDefinition([str(s) for s in get_settings().seasons.as_list()])


@asset(partitions_def=seasons, group_name="bronze", compute_kind="python")
def bronze_jolpica_results(context: AssetExecutionContext) -> MaterializeResult:
    pages = pipelines.ingest_jolpica(int(context.partition_key), "results")
    return MaterializeResult(metadata={"pages": pages})


@asset(partitions_def=seasons, group_name="bronze", compute_kind="python")
def bronze_fastf1_laps(context: AssetExecutionContext) -> MaterializeResult:
    season = int(context.partition_key)
    rows = 0
    for round_ in event_rounds(season):
        for session_type in get_settings().fastf1.session_types:
            rows += pipelines.ingest_fastf1_laps(season, round_, session_type)
    return MaterializeResult(metadata={"rows": rows})


@asset(partitions_def=seasons, group_name="silver", compute_kind="spark", deps=[bronze_fastf1_laps])
def silver_laps(context: AssetExecutionContext) -> MaterializeResult:
    report = pipelines.build_silver_laps(int(context.partition_key))
    return MaterializeResult(
        metadata={
            "rows_total": report.total_rows,
            "rows_quarantined": report.quarantined_rows,
            "quarantine_ratio": report.quarantine_ratio,
            **{f"dq_fail_{k}": v for k, v in report.failures.items()},
        }
    )


defs = Definitions(assets=[bronze_jolpica_results, bronze_fastf1_laps, silver_laps])
