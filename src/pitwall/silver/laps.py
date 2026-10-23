"""bronze.fastf1_laps -> silver.laps: typed, deduplicated, flagged."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from pitwall.quality import Expectation
from pitwall.silver.common import latest_per_key

NATURAL_KEY = ["season", "event_round", "session_type", "driver_number", "lap_number"]
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
ALL_COMPOUNDS = (*DRY_COMPOUNDS, "INTERMEDIATE", "WET")

EXPECTATIONS = [
    Expectation("natural_key_not_null", " AND ".join(f"{k} IS NOT NULL" for k in NATURAL_KEY)),
    Expectation("lap_number_positive", "lap_number >= 1"),
    Expectation("lap_time_plausible", "lap_time_s IS NULL OR lap_time_s BETWEEN 60 AND 300"),
    Expectation("tyre_life_positive", "tyre_life_laps IS NULL OR tyre_life_laps >= 1"),
    Expectation("lap_time_present", "lap_time_s IS NOT NULL", severity="warn"),
    Expectation(
        "compound_known",
        "compound IN (" + ", ".join(f"'{c}'" for c in ALL_COMPOUNDS) + ")",
        severity="warn",
    ),
]


def _seconds(col: str) -> Column:
    """FastF1 timedeltas land as float seconds; NaN means missing."""
    c = F.col(col).cast("double")
    return F.when(F.isnan(c), F.lit(None)).otherwise(c)


def transform_laps(bronze: DataFrame) -> DataFrame:
    typed = bronze.select(
        F.col("season").cast("int").alias("season"),
        F.col("event_round").cast("int").alias("event_round"),
        F.col("session_type"),
        F.col("driver").alias("driver_code"),
        F.col("driver_number").cast("int").alias("driver_number"),
        F.col("team").alias("team_name"),
        F.col("lap_number").cast("int").alias("lap_number"),
        F.col("stint").cast("int").alias("stint_number"),
        _seconds("lap_time").alias("lap_time_s"),
        _seconds("sector1_time").alias("sector1_s"),
        _seconds("sector2_time").alias("sector2_s"),
        _seconds("sector3_time").alias("sector3_s"),
        F.upper(F.col("compound")).alias("compound"),
        F.col("tyre_life").cast("int").alias("tyre_life_laps"),
        F.col("fresh_tyre").cast("boolean").alias("fresh_tyre"),
        _seconds("pit_in_time").isNotNull().alias("is_pit_in_lap"),
        _seconds("pit_out_time").isNotNull().alias("is_pit_out_lap"),
        F.col("track_status").cast("string").alias("track_status"),
        F.col("is_accurate").cast("boolean").alias("is_accurate"),
        F.col("deleted").cast("boolean").alias("is_deleted"),
        F.col("position").cast("int").alias("position_end_of_lap"),
        # Session time (s since session start) at the lap's start and at the line.
        _seconds("lap_start_time").alias("lap_start_time_s"),
        _seconds("time").alias("lap_end_time_s"),
        _seconds("speed_st").alias("speed_trap_kmh"),
        F.col("_ingested_at"),
        F.col("_batch_id").alias("_source_batch_id"),
    )
    # TrackStatus lists every status seen during the lap; "1" alone = fully green.
    typed = typed.withColumn("is_green_flag", F.col("track_status") == F.lit("1"))
    return latest_per_key(typed, NATURAL_KEY).withColumn("_updated_at", F.current_timestamp())
