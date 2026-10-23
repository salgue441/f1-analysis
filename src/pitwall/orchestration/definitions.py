"""Dagster asset graph (ADR-0004). Assets are thin wrappers over `pitwall.pipelines`.

No `from __future__ import annotations` here: Dagster inspects the context type hint at runtime.
"""

from dagster import (
    AssetExecutionContext,
    AssetsDefinition,
    AssetSelection,
    Definitions,
    MaterializeResult,
    RunRequest,
    ScheduleEvaluationContext,
    StaticPartitionsDefinition,
    asset,
    define_asset_job,
    schedule,
)

from pitwall import pipelines
from pitwall.config import get_settings
from pitwall.ingestion.fastf1_loader import event_rounds
from pitwall.quality import QualityReport

seasons = StaticPartitionsDefinition([str(s) for s in get_settings().seasons.as_list()])


def _dq_metadata(report: QualityReport) -> dict:
    return {
        "rows_total": report.total_rows,
        "rows_quarantined": report.quarantined_rows,
        "quarantine_ratio": report.quarantine_ratio,
        **{f"dq_fail_{k}": v for k, v in report.failures.items()},
    }


# --- bronze ----------------------------------------------------------------------------------


@asset(partitions_def=seasons, group_name="bronze", compute_kind="python")
def bronze_jolpica(context: AssetExecutionContext) -> MaterializeResult:
    """Season-level Jolpica endpoints: races, results, sprint, qualifying, entities."""
    season = int(context.partition_key)
    pages = {e: pipelines.ingest_jolpica(season, e) for e in pipelines.SEASON_ENDPOINTS}
    return MaterializeResult(metadata={f"pages_{k}": v for k, v in pages.items()})


@asset(partitions_def=seasons, group_name="bronze", compute_kind="python", deps=["silver_races"])
def bronze_jolpica_pitstops(context: AssetExecutionContext) -> MaterializeResult:
    """Pit stops are served per round, so this waits for the season's race list."""
    season = int(context.partition_key)
    pages = sum(
        pipelines.ingest_jolpica(season, "pitstops", r) for r in pipelines.season_rounds(season)
    )
    return MaterializeResult(metadata={"pages": pages})


@asset(partitions_def=seasons, group_name="bronze", compute_kind="python")
def bronze_fastf1_laps(context: AssetExecutionContext) -> MaterializeResult:
    season = int(context.partition_key)
    rows = 0
    for round_ in event_rounds(season):
        for session_type in get_settings().fastf1.session_types:
            rows += pipelines.ingest_fastf1_laps(season, round_, session_type)
    return MaterializeResult(metadata={"rows": rows})


# --- silver ----------------------------------------------------------------------------------


def silver_asset(asset_name: str, specs: list[str], dep: str) -> AssetsDefinition:
    """One silver asset; several specs can feed the same table (results + sprint)."""

    @asset(
        name=asset_name,
        partitions_def=seasons,
        group_name="silver",
        compute_kind="spark",
        deps=[dep],
    )
    def _silver(context: AssetExecutionContext) -> MaterializeResult:
        metadata: dict = {}
        for spec in specs:
            report = pipelines.build_silver(spec, int(context.partition_key))
            prefix = "" if len(specs) == 1 else f"{spec}_"
            metadata.update({prefix + k: v for k, v in _dq_metadata(report).items()})
        return MaterializeResult(metadata=metadata)

    return _silver


SILVER_ASSETS = [
    silver_asset("silver_races", ["races"], "bronze_jolpica"),
    silver_asset("silver_circuits", ["circuits"], "bronze_jolpica"),
    silver_asset("silver_drivers", ["drivers"], "bronze_jolpica"),
    silver_asset("silver_constructors", ["constructors"], "bronze_jolpica"),
    silver_asset("silver_results", ["results", "sprint_results"], "bronze_jolpica"),
    silver_asset("silver_qualifying", ["qualifying"], "bronze_jolpica"),
    silver_asset("silver_pit_stops", ["pit_stops"], "bronze_jolpica_pitstops"),
    silver_asset("silver_laps", ["laps"], "bronze_fastf1_laps"),
]


# --- gold ------------------------------------------------------------------------------------


@asset(
    partitions_def=seasons,
    group_name="gold",
    compute_kind="spark",
    deps=[
        "silver_races",
        "silver_circuits",
        "silver_drivers",
        "silver_constructors",
        "silver_results",
        "silver_qualifying",
    ],
)
def gold_dims(context: AssetExecutionContext) -> MaterializeResult:
    """All dimensions, rebuilt together from every season in silver.

    Partitioned only so one season job covers the whole graph; any partition rebuilds all.
    """
    return MaterializeResult(
        metadata={f"rows_{k}": v for k, v in pipelines.build_gold_dims().items()}
    )


@asset(partitions_def=seasons, group_name="gold", compute_kind="spark", deps=[gold_dims])
def gold_fact_session_result(context: AssetExecutionContext) -> MaterializeResult:
    rows = pipelines.build_fact_session_result(int(context.partition_key))
    return MaterializeResult(metadata={"rows": rows})


# --- jobs & schedules ------------------------------------------------------------------------

season_refresh = define_asset_job(
    "season_refresh",
    selection=AssetSelection.all(),
)


@schedule(job=season_refresh, cron_schedule="0 6 * * 1", execution_timezone="UTC")
def post_race_weekend(context: ScheduleEvaluationContext):
    """Monday 06:00 UTC: refresh the current season after a race weekend (SAD §7.1)."""
    when = context.scheduled_execution_time
    season = str(when.year)
    if season in seasons.get_partition_keys():
        yield RunRequest(partition_key=season, run_key=f"{season}-{when.date().isoformat()}")


defs = Definitions(
    assets=[
        bronze_jolpica,
        bronze_jolpica_pitstops,
        bronze_fastf1_laps,
        *SILVER_ASSETS,
        gold_dims,
        gold_fact_session_result,
    ],
    jobs=[season_refresh],
    schedules=[post_race_weekend],
)
