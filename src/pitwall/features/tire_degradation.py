"""feat_tire_degradation (RQ1): one row per clean green-flag race lap on dry tyres.

See docs/data-model.md#feat_tire_degradation-rq1. Every column is either known before the
race (scheduled laps, qualifying pace) or measured on the lap itself, so a lap's features
never depend on later laps (SAD §9.3; tested).
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pitwall.gold.lap_facts import clean_lap_flag

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
TRAFFIC_GAP_S = 1.5
# Laps slower than this multiple of the weekend's pole time are not pace laps (damage,
# cool-down, a spin that still passed the flag filters). Pole time is known before the race.
MAX_LAP_OVER_POLE = 1.15

COLUMNS = [
    "season",
    "event_round",
    "race_id",
    "circuit_id",
    "driver_id",
    "constructor_id",
    "lap_number",
    "stint_number",
    "compound",
    "tyre_life_laps",
    "fuel_proxy",
    "track_temp_c",
    "gap_to_car_ahead_s",
    "in_traffic",
    "lap_time_s",
]


def build_feat_tire_degradation(
    fact_lap: DataFrame,
    dim_session: DataFrame,
    fact_session_result: DataFrame,
    traffic_gap_s: float = TRAFFIC_GAP_S,
) -> DataFrame:
    races = dim_session.filter(F.col("session_type") == "R").select(
        "season",
        F.col("round").alias("event_round"),
        "circuit_id",
        "scheduled_laps",
    )
    pole = (
        fact_session_result.filter(F.col("session_type") == "Q")
        .groupBy("season", F.col("round").alias("event_round"))
        .agg(F.min("best_lap_time_s").alias("_pole_s"))
    )
    laps = (
        fact_lap.filter(F.col("session_type") == "R")
        .filter(clean_lap_flag())
        .filter(F.col("compound").isin(*DRY_COMPOUNDS))
        .filter(F.col("lap_number") > 1)
        .filter(F.col("tyre_life_laps") >= 1)
        .filter(F.col("track_temp_c").isNotNull())
        .join(races, ["season", "event_round"])
        .join(pole, ["season", "event_round"], "left")
        .filter(
            F.col("_pole_s").isNull()
            | (F.col("lap_time_s") <= MAX_LAP_OVER_POLE * F.col("_pole_s"))
        )
    )
    return laps.select(
        "season",
        "event_round",
        F.format_string("%d_%02d", "season", "event_round").alias("race_id"),
        "circuit_id",
        "driver_id",
        "constructor_id",
        "lap_number",
        "stint_number",
        "compound",
        "tyre_life_laps",
        # Laps still to run: a proxy for fuel mass on board at the start of the lap.
        (F.col("scheduled_laps") - F.col("lap_number")).alias("fuel_proxy"),
        "track_temp_c",
        "gap_to_car_ahead_s",
        F.coalesce(F.col("gap_to_car_ahead_s") < traffic_gap_s, F.lit(False)).alias("in_traffic"),
        "lap_time_s",
    )
