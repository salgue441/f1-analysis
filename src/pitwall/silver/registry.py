"""Silver table specs: which bronze table feeds which silver table, and how."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyspark.sql import DataFrame

from pitwall.quality import Expectation
from pitwall.silver import fastf1, jolpica, openf1
from pitwall.silver import laps as silver_laps

TELEMETRY_PARTITIONS = ["season", "event_round", "session_type"]


@dataclass(frozen=True)
class SilverSpec:
    name: str
    # One bronze table, or several passed to `transform` positionally (OpenF1 laps).
    bronze_table: str | tuple[str, ...]
    transform: Callable[..., DataFrame]
    natural_key: list[str]
    expectations: list[Expectation]
    # Season-scoped tables are partitioned by season; entity tables span all seasons.
    partition_by: list[str] | None = None
    # Column holding the season in bronze, used to load one season at a time.
    bronze_season_col: str = "season"
    # Telemetry replaces whole session partitions instead of MERGE: same idempotency, no
    # row-level matching over ~10^7 samples per season.
    replace_partitions: bool = False


SPECS: dict[str, SilverSpec] = {
    spec.name: spec
    for spec in [
        SilverSpec(
            "races",
            "jolpica_races",
            jolpica.transform_races,
            jolpica.RACES_KEY,
            jolpica.RACES_EXPECTATIONS,
            ["season"],
        ),
        SilverSpec(
            "circuits",
            "jolpica_circuits",
            jolpica.transform_circuits,
            jolpica.CIRCUITS_KEY,
            jolpica.CIRCUITS_EXPECTATIONS,
        ),
        SilverSpec(
            "drivers",
            "jolpica_drivers",
            jolpica.transform_drivers,
            jolpica.DRIVERS_KEY,
            jolpica.DRIVERS_EXPECTATIONS,
        ),
        SilverSpec(
            "constructors",
            "jolpica_constructors",
            jolpica.transform_constructors,
            jolpica.CONSTRUCTORS_KEY,
            jolpica.CONSTRUCTORS_EXPECTATIONS,
        ),
        SilverSpec(
            "results",
            "jolpica_results",
            jolpica.transform_results,
            jolpica.RESULTS_KEY,
            jolpica.RESULTS_EXPECTATIONS,
            ["season"],
        ),
        # Sprint results land in the same silver table, distinguished by session_type.
        SilverSpec(
            "sprint_results",
            "jolpica_sprint",
            jolpica.transform_sprint_results,
            jolpica.RESULTS_KEY,
            jolpica.RESULTS_EXPECTATIONS,
            ["season"],
        ),
        SilverSpec(
            "qualifying",
            "jolpica_qualifying",
            jolpica.transform_qualifying,
            jolpica.QUALIFYING_KEY,
            jolpica.QUALIFYING_EXPECTATIONS,
            ["season"],
        ),
        SilverSpec(
            "pit_stops",
            "jolpica_pitstops",
            jolpica.transform_pit_stops,
            jolpica.PIT_STOPS_KEY,
            jolpica.PIT_STOPS_EXPECTATIONS,
            ["season"],
        ),
        SilverSpec(
            "fastf1_laps",
            "fastf1_laps",
            silver_laps.transform_laps,
            silver_laps.NATURAL_KEY,
            silver_laps.EXPECTATIONS,
            ["season"],
        ),
        SilverSpec(
            "fastf1_weather",
            "fastf1_weather",
            fastf1.transform_weather,
            fastf1.WEATHER_KEY,
            fastf1.WEATHER_EXPECTATIONS,
            ["season"],
        ),
        SilverSpec(
            "fastf1_race_control",
            "fastf1_track_status",
            fastf1.transform_race_control,
            fastf1.RACE_CONTROL_KEY,
            fastf1.RACE_CONTROL_EXPECTATIONS,
            ["season"],
        ),
        SilverSpec(
            "fastf1_telemetry_car",
            "fastf1_car_data",
            fastf1.transform_car_data,
            fastf1.TELEMETRY_KEY,
            fastf1.CAR_EXPECTATIONS,
            TELEMETRY_PARTITIONS,
            replace_partitions=True,
        ),
        SilverSpec(
            "fastf1_telemetry_pos",
            "fastf1_pos_data",
            fastf1.transform_pos_data,
            fastf1.TELEMETRY_KEY,
            fastf1.POS_EXPECTATIONS,
            TELEMETRY_PARTITIONS,
            replace_partitions=True,
        ),
    ]
}

OPENF1_SPECS = [
    SilverSpec(
        "laps",
        tuple(
            f"openf1_{e}" for e in ["laps", "stints", "pit", "position", "race_control", "drivers"]
        ),
        openf1.transform_laps,
        silver_laps.NATURAL_KEY,
        silver_laps.EXPECTATIONS,
        ["season"],
    ),
    SilverSpec(
        "weather",
        "openf1_weather",
        openf1.transform_weather,
        fastf1.WEATHER_KEY,
        fastf1.WEATHER_EXPECTATIONS,
        ["season"],
    ),
    SilverSpec(
        "race_control",
        "openf1_race_control",
        openf1.transform_race_control,
        fastf1.RACE_CONTROL_KEY,
        fastf1.RACE_CONTROL_EXPECTATIONS,
        ["season"],
    ),
    SilverSpec(
        "telemetry_car",
        "openf1_car_data",
        openf1.transform_car_data,
        fastf1.TELEMETRY_KEY,
        fastf1.CAR_EXPECTATIONS,
        TELEMETRY_PARTITIONS,
        replace_partitions=True,
    ),
    SilverSpec(
        "telemetry_pos",
        "openf1_location",
        openf1.transform_pos_data,
        fastf1.TELEMETRY_KEY,
        fastf1.POS_EXPECTATIONS,
        TELEMETRY_PARTITIONS,
        replace_partitions=True,
    ),
]
SPECS.update({spec.name: spec for spec in OPENF1_SPECS})

# Specs that write into a differently named silver table. The fastf1_* specs feed the same
# silver tables as OpenF1 (identical schemas), for when FastF1 is reachable again (ADR-0006).
TARGET_TABLE = {
    "sprint_results": "results",
    "fastf1_laps": "laps",
    "fastf1_weather": "weather",
    "fastf1_race_control": "race_control",
    "fastf1_telemetry_car": "telemetry_car",
    "fastf1_telemetry_pos": "telemetry_pos",
}


def target_table(spec: SilverSpec) -> str:
    return TARGET_TABLE.get(spec.name, spec.name)
