"""Lap-level gold facts: fact_lap, fact_stint, fact_pit_stop, fact_telemetry_sample."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from pitwall.asof import asof_backward
from pitwall.gold.dims import COMPOUNDS
from pitwall.gold.keys import surrogate_key

SESSION = ["season", "event_round", "session_type"]
NEUTRALIZED_CODES = ("4", "5", "6", "7")  # SC, red flag, VSC, VSC ending
KNOWN_COMPOUNDS = [c for c, _, _ in COMPOUNDS]


def _neutralized(track_status: str) -> Column:
    """True if any neutralization code appears in a lap's TrackStatus string (e.g. '124')."""
    return F.col(track_status).rlike("[" + "".join(NEUTRALIZED_CODES) + "]")


def clean_lap_flag() -> Column:
    """A lap usable for pace analysis: timed, green, accurate, not in/out of the pits."""
    return (
        F.col("lap_time_s").isNotNull()
        & F.col("is_green_flag")
        & F.coalesce(F.col("is_accurate"), F.lit(False))
        & ~F.col("is_pit_in_lap")
        & ~F.col("is_pit_out_lap")
        & ~F.coalesce(F.col("is_deleted"), F.lit(False))
    )


def with_gap_to_car_ahead(laps: DataFrame) -> DataFrame:
    """Gap (s) at the line to the car one position ahead on the same lap."""
    # Position feeds can briefly report two cars in one position; take the earlier car so
    # the join never duplicates a lap.
    ahead = laps.groupBy(
        *SESSION,
        "lap_number",
        (F.col("position_end_of_lap") + 1).alias("position_end_of_lap"),
    ).agg(F.min("lap_end_time_s").alias("_ahead_end_s"))
    return (
        laps.join(ahead, [*SESSION, "lap_number", "position_end_of_lap"], "left")
        .withColumn("gap_to_car_ahead_s", F.col("lap_end_time_s") - F.col("_ahead_end_s"))
        .drop("_ahead_end_s")
    )


def build_fact_lap(laps: DataFrame, weather: DataFrame, driver_map: DataFrame) -> DataFrame:
    """Grain: driver x lap x session.

    Weather is attached as of the lap start (backward-looking: the latest sample at or
    before `lap_start_time_s`), so features built from this table can't see the future.
    """
    mapped = laps.join(
        driver_map.select(
            "season",
            F.col("round").alias("event_round"),
            "driver_number",
            "driver_id",
            "constructor_id",
        ),
        ["season", "event_round", "driver_number"],
        "left",
    )
    with_weather = asof_backward(
        with_gap_to_car_ahead(mapped),
        weather,
        by=SESSION,
        left_time="lap_start_time_s",
        right_time="session_time_s",
        values=["track_temp_c", "air_temp_c", "is_raining"],
    )
    known = F.col("compound").isin(*KNOWN_COMPOUNDS)
    with_weather = with_weather.withColumn(
        "compound", F.when(known, F.col("compound")).otherwise(F.lit("UNKNOWN"))
    )
    return with_weather.select(
        surrogate_key(*SESSION, "driver_number", "lap_number").alias("lap_key"),
        surrogate_key(*SESSION).alias("session_key"),
        surrogate_key("driver_id").alias("driver_key"),
        surrogate_key("constructor_id").alias("constructor_key"),
        surrogate_key("compound").alias("compound_key"),
        *SESSION,
        "driver_number",
        "driver_id",
        "constructor_id",
        "lap_number",
        "stint_number",
        "compound",
        "tyre_life_laps",
        "fresh_tyre",
        "lap_time_s",
        "sector1_s",
        "sector2_s",
        "sector3_s",
        "lap_start_time_s",
        "lap_end_time_s",
        "position_end_of_lap",
        "gap_to_car_ahead_s",
        "speed_trap_kmh",
        "is_pit_in_lap",
        "is_pit_out_lap",
        "is_green_flag",
        _neutralized("track_status").alias("is_neutralized"),
        "track_status",
        "is_accurate",
        "is_deleted",
        "track_temp_c",
        "air_temp_c",
        "is_raining",
        "_source_batch_id",
        F.current_timestamp().alias("_updated_at"),
    )


def build_fact_stint(fact_lap: DataFrame) -> DataFrame:
    """Grain: driver x stint x session. Pace stats use clean laps only.

    `raw_deg_slope_s_per_lap` is the OLS slope of lap time on tyre age within the stint,
    not fuel corrected (fuel burn makes it an underestimate). RQ1 does the careful version.
    """
    keys = [*SESSION, "driver_number", "stint_number"]
    clean = clean_lap_flag()
    t, y = F.col("tyre_life_laps").cast("double"), F.col("lap_time_s")
    clean_t, clean_y = F.when(clean, t), F.when(clean, y)
    stints = (
        fact_lap.filter(F.col("stint_number").isNotNull())
        .groupBy(*keys)
        .agg(
            F.first("session_key").alias("session_key"),
            F.first("driver_key").alias("driver_key"),
            F.first("constructor_key").alias("constructor_key"),
            F.first("driver_id").alias("driver_id"),
            # The compound can only be UNKNOWN on some laps of a stint; prefer a known one.
            F.max(F.when(F.col("compound") != "UNKNOWN", F.col("compound"))).alias("compound"),
            F.min("lap_number").alias("start_lap"),
            F.max("lap_number").alias("end_lap"),
            F.count(F.lit(1)).alias("laps"),
            F.min("tyre_life_laps").alias("start_tyre_life_laps"),
            # max, not first: first() depends on row order, and rebuilds must be identical.
            F.max("fresh_tyre").alias("fresh_tyre"),
            F.count(clean_y).alias("clean_laps"),
            # Rounded: floating-point sums depend on aggregation order, and rebuilds must be
            # bit-identical for the fingerprint check (SAD §11).
            F.round(F.mean(clean_y), 6).alias("mean_clean_lap_s"),
            F.percentile_approx(clean_y, 0.5).alias("median_clean_lap_s"),
            F.round(F.covar_samp(clean_t, clean_y) / F.var_samp(clean_t), 6).alias(
                "raw_deg_slope_s_per_lap"
            ),
        )
    )
    stints = stints.withColumn("compound", F.coalesce(F.col("compound"), F.lit("UNKNOWN")))
    return stints.select(
        surrogate_key(*keys).alias("stint_key"),
        surrogate_key("compound").alias("compound_key"),
        "*",
        F.current_timestamp().alias("_updated_at"),
    )


def build_fact_pit_stop(
    pit_stops: DataFrame, fact_lap: DataFrame, driver_map: DataFrame
) -> DataFrame:
    """Grain: driver x stop x race. Jolpica gives the stop; FastF1 laps add context.

    `pit_lane_s` is Jolpica's pit-lane entry-to-exit time. Stationary time is not published
    by either source, so it is not modeled. `under_neutralization` is true when the in-lap
    ran under SC/VSC/red flag, which makes the stop much cheaper.
    """
    race_laps = fact_lap.filter(F.col("session_type") == "R")
    in_lap = race_laps.select(
        "season",
        F.col("event_round").alias("round"),
        "driver_id",
        F.col("lap_number").alias("lap"),
        F.col("compound").alias("compound_before"),
        F.col("tyre_life_laps").alias("tyre_life_before"),
        F.col("is_neutralized").alias("under_neutralization"),
        F.col("position_end_of_lap").alias("position_before"),
    )
    out_lap = race_laps.select(
        "season",
        F.col("event_round").alias("round"),
        "driver_id",
        (F.col("lap_number") - 1).alias("lap"),
        F.col("compound").alias("compound_after"),
        F.col("position_end_of_lap").alias("position_after"),
    )
    numbers = driver_map.select("season", "round", "driver_id", "driver_number")
    return (
        pit_stops.join(numbers, ["season", "round", "driver_id"], "left")
        .join(in_lap, ["season", "round", "driver_id", "lap"], "left")
        .join(out_lap, ["season", "round", "driver_id", "lap"], "left")
        .select(
            surrogate_key("season", "round", "driver_id", "stop_number").alias("pit_stop_key"),
            surrogate_key("season", "round", F.lit("R").alias("session_type")).alias("session_key"),
            surrogate_key("driver_id").alias("driver_key"),
            "season",
            "round",
            "driver_id",
            "driver_number",
            "stop_number",
            "lap",
            F.col("duration_s").alias("pit_lane_s"),
            "compound_before",
            "compound_after",
            "tyre_life_before",
            "position_before",
            "position_after",
            F.coalesce(F.col("under_neutralization"), F.lit(False)).alias("under_neutralization"),
            "_source_batch_id",
            F.current_timestamp().alias("_updated_at"),
        )
    )


def build_fact_telemetry_sample(car: DataFrame, pos: DataFrame) -> DataFrame:
    """Grain: driver x car-data sample x session, with the latest position sample attached.

    Car data (~4 Hz) and position data (~4 Hz, not aligned) are joined backward as-of on
    session time. `distance_m` integrates speed over time per driver and session.
    """
    by = [*SESSION, "driver_number"]
    joined = asof_backward(
        car,
        pos.select(*by, F.col("session_time_s").alias("_pos_t"), "x_m", "y_m", "z_m"),
        by=by,
        left_time="session_time_s",
        right_time="_pos_t",
        values=["x_m", "y_m", "z_m"],
    )
    in_order = Window.partitionBy(*by).orderBy("session_time_s")
    dt_s = F.col("session_time_s") - F.lag("session_time_s").over(in_order)
    step_m = F.coalesce(dt_s * F.col("speed_kmh") / 3.6, F.lit(0.0))
    return (
        joined.withColumn("_step_m", step_m)
        .withColumn(
            "distance_m",
            F.sum("_step_m").over(in_order.rowsBetween(Window.unboundedPreceding, 0)),
        )
        .drop("_step_m")
        .select(
            surrogate_key(*SESSION).alias("session_key"),
            *by,
            "session_time_s",
            "sample_utc",
            "speed_kmh",
            "rpm",
            "gear",
            "throttle_pct",
            "is_braking",
            "drs",
            "x_m",
            "y_m",
            "z_m",
            "distance_m",
            "_source_batch_id",
            F.current_timestamp().alias("_updated_at"),
        )
    )
