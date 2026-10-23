"""Dagster asset graph (ADR-0004). Assets are thin wrappers over `pitwall.pipelines`.

Two partition sets: `seasons` (2018+, results from Jolpica) and `lap_seasons` (2023+, lap
data from OpenF1, ADR-0006). Lap assets depend on season assets one-to-one by key.

No `from __future__ import annotations` here: Dagster inspects the context type hint at runtime.
"""

from collections.abc import Callable

from dagster import (
    AssetExecutionContext,
    AssetsDefinition,
    AssetSelection,
    Definitions,
    MaterializeResult,
    RunRequest,
    ScheduleDefinition,
    ScheduleEvaluationContext,
    StaticPartitionsDefinition,
    asset,
    define_asset_job,
    schedule,
)

from pitwall import pipelines
from pitwall.config import get_settings
from pitwall.ingestion import openf1
from pitwall.quality import QualityReport

seasons = StaticPartitionsDefinition([str(s) for s in get_settings().seasons.as_list()])
lap_seasons = StaticPartitionsDefinition([str(s) for s in get_settings().lap_seasons.as_list()])


def _dq_metadata(report: QualityReport) -> dict:
    return {
        "rows_total": report.total_rows,
        "rows_quarantined": report.quarantined_rows,
        "quarantine_ratio": report.quarantine_ratio,
        **{f"dq_fail_{k}": v for k, v in report.failures.items()},
    }


def silver_asset(
    asset_name: str,
    specs: list[str],
    dep: str,
    partitions: StaticPartitionsDefinition = seasons,
    group: str = "silver",
) -> AssetsDefinition:
    """One silver asset; several specs can feed the same table (results + sprint)."""

    @asset(
        name=asset_name,
        partitions_def=partitions,
        group_name=group,
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


def season_asset(
    name: str,
    build: Callable[[int], int],
    deps: list[str],
    partitions: StaticPartitionsDefinition = lap_seasons,
    group: str = "gold",
) -> AssetsDefinition:
    @asset(name=name, partitions_def=partitions, group_name=group, compute_kind="spark", deps=deps)
    def _built(context: AssetExecutionContext) -> MaterializeResult:
        return MaterializeResult(metadata={"rows": build(int(context.partition_key))})

    return _built


# --- results history (Jolpica, 2018+) --------------------------------------------------------


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


RESULT_SILVER_ASSETS = [
    silver_asset("silver_races", ["races"], "bronze_jolpica"),
    silver_asset("silver_circuits", ["circuits"], "bronze_jolpica"),
    silver_asset("silver_drivers", ["drivers"], "bronze_jolpica"),
    silver_asset("silver_constructors", ["constructors"], "bronze_jolpica"),
    silver_asset("silver_results", ["results", "sprint_results"], "bronze_jolpica"),
    silver_asset("silver_qualifying", ["qualifying"], "bronze_jolpica"),
    silver_asset("silver_pit_stops", ["pit_stops"], "bronze_jolpica_pitstops"),
]


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


RESULT_GOLD_ASSETS = [
    season_asset(
        "gold_fact_session_result",
        pipelines.build_fact_session_result,
        ["gold_dims"],
        partitions=seasons,
    ),
    season_asset(
        "silver_driver_id_map",
        pipelines.build_driver_id_map,
        ["silver_results", "silver_qualifying", "silver_drivers"],
        partitions=seasons,
        group="silver",
    ),
]


# --- laps, tyres, weather (OpenF1, 2023+) ----------------------------------------------------


@asset(
    partitions_def=lap_seasons, group_name="bronze", compute_kind="python", deps=["silver_races"]
)
def bronze_openf1(context: AssetExecutionContext) -> MaterializeResult:
    """Laps, stints, pit, position, race control, weather and drivers for every race."""
    written = pipelines.ingest_openf1_season(int(context.partition_key))
    return MaterializeResult(metadata={f"rows_{k}": v for k, v in written.items()})


LAP_ASSETS = [
    *[
        silver_asset(f"silver_{name}", [name], "bronze_openf1", lap_seasons)
        for name in ["laps", "weather", "race_control"]
    ],
    season_asset(
        "gold_fact_lap",
        pipelines.build_fact_lap,
        ["silver_laps", "silver_weather", "silver_driver_id_map", "gold_dims"],
    ),
    season_asset("gold_fact_stint", pipelines.build_fact_stint, ["gold_fact_lap"]),
    season_asset(
        "gold_fact_pit_stop", pipelines.build_fact_pit_stop, ["gold_fact_lap", "silver_pit_stops"]
    ),
]


# --- telemetry (large; its own job, run on demand) -------------------------------------------


@asset(
    partitions_def=lap_seasons, group_name="telemetry", compute_kind="python", deps=["silver_races"]
)
def bronze_openf1_telemetry(context: AssetExecutionContext) -> MaterializeResult:
    written = pipelines.ingest_openf1_season(
        int(context.partition_key), endpoints=openf1.TELEMETRY_ENDPOINTS
    )
    return MaterializeResult(metadata={f"rows_{k}": v for k, v in written.items()})


TELEMETRY_ASSETS = [
    silver_asset(
        "silver_telemetry_car",
        ["telemetry_car"],
        "bronze_openf1_telemetry",
        lap_seasons,
        "telemetry",
    ),
    silver_asset(
        "silver_telemetry_pos",
        ["telemetry_pos"],
        "bronze_openf1_telemetry",
        lap_seasons,
        "telemetry",
    ),
    season_asset(
        "gold_fact_telemetry_sample",
        pipelines.build_fact_telemetry_sample,
        ["silver_telemetry_car", "silver_telemetry_pos"],
        group="telemetry",
    ),
]


@asset(group_name="maintenance", compute_kind="delta")
def lakehouse_maintenance() -> MaterializeResult:
    """OPTIMIZE (Z-order where configured) + VACUUM every table (SAD §6.3)."""
    report = pipelines.maintain()
    return MaterializeResult(
        metadata={f"files_{k}": v["after"]["files"] for k, v in report.items()}
    )


# --- jobs & schedules ------------------------------------------------------------------------

RESULT_ASSETS = [
    bronze_jolpica,
    bronze_jolpica_pitstops,
    *RESULT_SILVER_ASSETS,
    gold_dims,
    *RESULT_GOLD_ASSETS,
]

results_refresh = define_asset_job(
    "results_refresh", selection=AssetSelection.assets(*RESULT_ASSETS)
)
laps_refresh = define_asset_job(
    "laps_refresh", selection=AssetSelection.assets(bronze_openf1, *LAP_ASSETS)
)
telemetry_refresh = define_asset_job(
    "telemetry_refresh",
    selection=AssetSelection.assets(bronze_openf1_telemetry, *TELEMETRY_ASSETS),
)
maintenance_job = define_asset_job(
    "maintenance", selection=AssetSelection.assets(lakehouse_maintenance)
)


def _current_season_request(context: ScheduleEvaluationContext, keys: list[str]):
    when = context.scheduled_execution_time
    season = str(when.year)
    if season in keys:
        yield RunRequest(partition_key=season, run_key=f"{season}-{when.date().isoformat()}")


@schedule(job=results_refresh, cron_schedule="0 6 * * 1", execution_timezone="UTC")
def results_after_race_weekend(context: ScheduleEvaluationContext):
    """Monday 06:00 UTC: refresh the current season after a race weekend (SAD §7.1)."""
    yield from _current_season_request(context, seasons.get_partition_keys())


@schedule(job=laps_refresh, cron_schedule="0 8 * * 1", execution_timezone="UTC")
def laps_after_race_weekend(context: ScheduleEvaluationContext):
    """Monday 08:00 UTC, after results (driver_id_map and dims must be fresh)."""
    yield from _current_season_request(context, lap_seasons.get_partition_keys())


weekly_maintenance = ScheduleDefinition(
    job=maintenance_job, cron_schedule="0 4 * * 0", execution_timezone="UTC"
)


defs = Definitions(
    assets=[
        *RESULT_ASSETS,
        bronze_openf1,
        *LAP_ASSETS,
        bronze_openf1_telemetry,
        *TELEMETRY_ASSETS,
        lakehouse_maintenance,
    ],
    jobs=[results_refresh, laps_refresh, telemetry_refresh, maintenance_job],
    schedules=[results_after_race_weekend, laps_after_race_weekend, weekly_maintenance],
)
