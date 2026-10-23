"""Silver table specs: which bronze table feeds which silver table, and how."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyspark.sql import DataFrame

from pitwall.quality import Expectation
from pitwall.silver import jolpica
from pitwall.silver import laps as silver_laps


@dataclass(frozen=True)
class SilverSpec:
    name: str
    bronze_table: str
    transform: Callable[[DataFrame], DataFrame]
    natural_key: list[str]
    expectations: list[Expectation]
    # Season-scoped tables are partitioned by season; entity tables span all seasons.
    partition_by: list[str] | None = None
    # Column holding the season in bronze, used to load one season at a time.
    bronze_season_col: str = "season"


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
            "laps",
            "fastf1_laps",
            silver_laps.transform_laps,
            silver_laps.NATURAL_KEY,
            silver_laps.EXPECTATIONS,
            ["season"],
        ),
    ]
}

# Specs that write into a differently named silver table.
TARGET_TABLE = {"sprint_results": "results"}


def target_table(spec: SilverSpec) -> str:
    return TARGET_TABLE.get(spec.name, spec.name)
