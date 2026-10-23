"""bronze.openf1_* -> silver laps, weather, race_control and telemetry (ADR-0006).

Output schemas are identical to the FastF1 transforms (`silver.laps`, `silver.fastf1`), so
gold doesn't know which source fed silver. Where OpenF1 lacks a FastF1 field it is derived:

- session time: seconds since the session's scheduled start (`session_start_utc`);
- track status per lap: status codes of every race-control period overlapping the lap;
- `is_accurate`: lap has a duration and all three sectors and isn't a pit in/out lap;
- `is_deleted`: parsed from race-control "... LAP n DELETED" messages.

Bronze rows hold one API response each (`payload`, a JSON array) for one session.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from pitwall.asof import asof_backward
from pitwall.silver.common import latest_per_key
from pitwall.silver.fastf1 import NEUTRALIZED_STATUSES, SESSION_KEY, TRACK_STATUS_LABELS

SCHEMAS = {
    "drivers": "driver_number int, name_acronym string, team_name string",
    "laps": (
        "driver_number int, lap_number int, date_start string, lap_duration double, "
        "duration_sector_1 double, duration_sector_2 double, duration_sector_3 double, "
        "st_speed double, is_pit_out_lap boolean"
    ),
    "stints": (
        "driver_number int, stint_number int, lap_start int, lap_end int, compound string, "
        "tyre_age_at_start int"
    ),
    "pit": "driver_number int, lap_number int, pit_duration double, date string",
    "position": "driver_number int, date string, position int",
    "race_control": (
        "date string, lap_number int, category string, flag string, scope string, "
        "driver_number int, message string"
    ),
    "weather": (
        "date string, air_temperature double, track_temperature double, humidity double, "
        "pressure double, rainfall double, wind_speed double, wind_direction double"
    ),
    "car_data": (
        "driver_number int, date string, speed double, rpm double, n_gear int, throttle double, "
        "brake double, drs int"
    ),
    "location": "driver_number int, date string, x double, y double, z double",
}

# "CAR 44 (HAM) TIME 1:32.123 DELETED - TRACK LIMITS AT TURN 4 LAP 23 15:04:22"
_DELETED = r"CAR (\d+) .*DELETED.* LAP (\d+)"


def explode_payload(bronze: DataFrame, endpoint: str) -> DataFrame:
    """One row per API record, with the session columns and lineage of its bronze batch."""
    latest = bronze.groupBy(*SESSION_KEY).agg(F.max("_ingested_at").alias("_latest"))
    fresh = bronze.join(F.broadcast(latest), SESSION_KEY).filter(
        F.col("_ingested_at") == F.col("_latest")
    )
    schema = _ddl_struct_array(SCHEMAS[endpoint])
    return (
        fresh.select(
            F.col("season").cast("int").alias("season"),
            F.col("event_round").cast("int").alias("event_round"),
            "session_type",
            F.col("session_start_utc").cast("timestamp").alias("session_start_utc"),
            F.explode(F.from_json("payload", schema)).alias("r"),
            "_ingested_at",
            F.col("_batch_id").alias("_source_batch_id"),
        )
        .select("*", "r.*")
        .drop("r")
    )


def _ddl_struct_array(columns: str) -> str:
    fields = ", ".join(
        f"{name}:{dtype}" for name, dtype in (c.strip().split(" ") for c in columns.split(","))
    )
    return f"array<struct<{fields}>>"


def session_seconds(ts_col: str | Column) -> Column:
    """ISO timestamp -> seconds since the session's scheduled start."""
    ts = F.col(ts_col) if isinstance(ts_col, str) else ts_col
    return F.to_timestamp(ts).cast("double") - F.col("session_start_utc").cast("double")


# --- race control -> track status periods ------------------------------------------------------


def status_events(race_control: DataFrame) -> DataFrame:
    """Race-control messages that change the track status, coded like FastF1's TrackStatus."""
    msg, flag, scope = F.upper("message"), F.col("flag"), F.col("scope")
    track_flag = F.col("category") == "Flag"
    status = (
        F.when(
            (F.col("category") == "SafetyCar") & msg.contains("VIRTUAL SAFETY CAR DEPLOYED"), "6"
        )
        .when((F.col("category") == "SafetyCar") & msg.contains("VIRTUAL SAFETY CAR ENDING"), "7")
        .when((F.col("category") == "SafetyCar") & msg.contains("SAFETY CAR DEPLOYED"), "4")
        .when(track_flag & (scope == "Track") & (flag == "RED"), "5")
        .when(track_flag & (scope == "Track") & flag.isin("GREEN", "CLEAR"), "1")
    )
    return race_control.select(
        *SESSION_KEY,
        "session_start_utc",
        session_seconds("date").alias("start_s"),
        status.alias("status"),
        "_ingested_at",
        "_source_batch_id",
    ).filter(F.col("status").isNotNull())


def transform_race_control(bronze_race_control: DataFrame) -> DataFrame:
    """Same schema as silver.fastf1.transform_race_control: [start_s, end_s) periods."""
    events = status_events(explode_payload(bronze_race_control, "race_control"))
    in_session = Window.partitionBy(*SESSION_KEY).orderBy("start_s")
    labels = F.create_map(*[F.lit(x) for kv in TRACK_STATUS_LABELS.items() for x in kv])
    changes = (
        latest_per_key(events, [*SESSION_KEY, "start_s"])
        .withColumn("_prev", F.lag("status").over(in_session))
        # Repeated GREEN/CLEAR messages are not changes.
        .filter(F.col("_prev").isNull() | (F.col("_prev") != F.col("status")))
        .drop("_prev", "session_start_utc")
    )
    return (
        changes.withColumn("end_s", F.lead("start_s").over(in_session))
        .withColumn("status_label", labels[F.col("status")])
        .withColumn("is_neutralized", F.col("status").isin(*NEUTRALIZED_STATUSES))
        .withColumn("_updated_at", F.current_timestamp())
        .select(
            *SESSION_KEY,
            "start_s",
            "status",
            "_ingested_at",
            "_source_batch_id",
            "end_s",
            "status_label",
            "is_neutralized",
            "_updated_at",
        )
    )


# --- laps ------------------------------------------------------------------------------------


def _lap_track_status(laps: DataFrame, periods: DataFrame) -> DataFrame:
    """Concatenate (sorted) status codes of every period overlapping each lap, e.g. '14'."""
    lap_id = [*SESSION_KEY, "driver_number", "lap_number"]
    p = periods.select(
        *SESSION_KEY, F.col("start_s").alias("_ps"), F.col("end_s").alias("_pe"), "status"
    )
    overlap = (F.col("_ps") < F.col("lap_end_time_s")) & (
        F.col("_pe").isNull() | (F.col("_pe") > F.col("lap_start_time_s"))
    )
    codes = (
        laps.select(*lap_id, "lap_start_time_s", "lap_end_time_s")
        .join(p, SESSION_KEY)
        .filter(overlap)
        .groupBy(*lap_id)
        .agg(F.array_join(F.array_sort(F.collect_set("status")), "").alias("track_status"))
    )
    # No period overlaps a lap before the first message of the race: treat as green.
    return laps.join(codes, lap_id, "left").withColumn(
        "track_status", F.coalesce(F.col("track_status"), F.lit("1"))
    )


def transform_laps(
    bronze_laps: DataFrame,
    bronze_stints: DataFrame,
    bronze_pit: DataFrame,
    bronze_position: DataFrame,
    bronze_race_control: DataFrame,
    bronze_drivers: DataFrame,
) -> DataFrame:
    """OpenF1 -> silver.laps with exactly the FastF1-based schema."""
    laps = explode_payload(bronze_laps, "laps").select(
        *SESSION_KEY,
        "session_start_utc",
        "driver_number",
        "lap_number",
        F.col("lap_duration").alias("lap_time_s"),
        F.col("duration_sector_1").alias("sector1_s"),
        F.col("duration_sector_2").alias("sector2_s"),
        F.col("duration_sector_3").alias("sector3_s"),
        session_seconds("date_start").alias("lap_start_time_s"),
        F.col("st_speed").alias("speed_trap_kmh"),
        F.coalesce(F.col("is_pit_out_lap"), F.lit(False)).alias("is_pit_out_lap"),
        "_ingested_at",
        "_source_batch_id",
    )
    laps = latest_per_key(laps, [*SESSION_KEY, "driver_number", "lap_number"]).withColumn(
        "lap_end_time_s", F.col("lap_start_time_s") + F.col("lap_time_s")
    )

    stints = explode_payload(bronze_stints, "stints").select(
        *[F.col(k).alias(f"_s_{k}") for k in [*SESSION_KEY, "driver_number"]],
        "stint_number",
        "lap_start",
        "lap_end",
        F.upper("compound").alias("compound"),
        "tyre_age_at_start",
    )
    # Range condition inside a left join: a lap outside every stint keeps NULL tyre columns.
    in_stint = [F.col(k) == F.col(f"_s_{k}") for k in [*SESSION_KEY, "driver_number"]]
    in_stint.append(F.col("lap_number").between(F.col("lap_start"), F.col("lap_end")))
    laps = (
        laps.join(stints, in_stint, "left")
        .withColumn(
            "tyre_life_laps",
            F.col("tyre_age_at_start") + F.col("lap_number") - F.col("lap_start") + 1,
        )
        .withColumn("fresh_tyre", F.col("tyre_age_at_start") == 0)
        .drop(*[c for c in stints.columns if c.startswith("_s_")])
        # Overlapping stint records would duplicate a lap; keep the later stint.
        .withColumn(
            "_rn",
            F.row_number().over(
                Window.partitionBy(*SESSION_KEY, "driver_number", "lap_number").orderBy(
                    F.col("stint_number").desc_nulls_last()
                )
            ),
        )
        .filter(F.col("_rn") == 1)
        .drop("_rn", "lap_start", "lap_end", "tyre_age_at_start")
    )

    pit_laps = (
        explode_payload(bronze_pit, "pit")
        .select(*SESSION_KEY, "driver_number", "lap_number", F.lit(True).alias("is_pit_in_lap"))
        .dropDuplicates([*SESSION_KEY, "driver_number", "lap_number"])
    )
    laps = laps.join(pit_laps, [*SESSION_KEY, "driver_number", "lap_number"], "left").withColumn(
        "is_pit_in_lap", F.coalesce(F.col("is_pit_in_lap"), F.lit(False))
    )

    positions = explode_payload(bronze_position, "position").select(
        *SESSION_KEY,
        "driver_number",
        session_seconds("date").alias("_pos_t"),
        F.col("position").alias("position_end_of_lap"),
    )
    laps = asof_backward(
        laps,
        positions,
        by=[*SESSION_KEY, "driver_number"],
        left_time="lap_end_time_s",
        right_time="_pos_t",
        values=["position_end_of_lap"],
    )

    race_control = explode_payload(bronze_race_control, "race_control")
    laps = _lap_track_status(laps, transform_race_control(bronze_race_control))
    deleted = (
        race_control.filter(F.upper("message").rlike(_DELETED))
        .select(
            *SESSION_KEY,
            F.regexp_extract(F.upper("message"), _DELETED, 1).cast("int").alias("driver_number"),
            F.regexp_extract(F.upper("message"), _DELETED, 2).cast("int").alias("lap_number"),
            F.lit(True).alias("is_deleted"),
        )
        .dropDuplicates()
    )
    laps = laps.join(deleted, [*SESSION_KEY, "driver_number", "lap_number"], "left")

    drivers = (
        explode_payload(bronze_drivers, "drivers")
        .select(*SESSION_KEY, "driver_number", "name_acronym", "team_name")
        .dropDuplicates([*SESSION_KEY, "driver_number"])
    )
    laps = laps.join(drivers, [*SESSION_KEY, "driver_number"], "left")

    accurate = (
        F.col("lap_time_s").isNotNull()
        & F.col("sector1_s").isNotNull()
        & F.col("sector2_s").isNotNull()
        & F.col("sector3_s").isNotNull()
        & ~F.col("is_pit_in_lap")
        & ~F.col("is_pit_out_lap")
    )
    return laps.select(
        "season",
        "event_round",
        "session_type",
        F.col("name_acronym").alias("driver_code"),
        "driver_number",
        "team_name",
        "lap_number",
        "stint_number",
        "lap_time_s",
        "sector1_s",
        "sector2_s",
        "sector3_s",
        "compound",
        F.col("tyre_life_laps").cast("int").alias("tyre_life_laps"),
        "fresh_tyre",
        "is_pit_in_lap",
        "is_pit_out_lap",
        "track_status",
        accurate.alias("is_accurate"),
        F.coalesce(F.col("is_deleted"), F.lit(False)).alias("is_deleted"),
        "position_end_of_lap",
        "lap_start_time_s",
        "lap_end_time_s",
        "speed_trap_kmh",
        "_ingested_at",
        "_source_batch_id",
        (F.col("track_status") == "1").alias("is_green_flag"),
        F.current_timestamp().alias("_updated_at"),
    )


# --- weather & telemetry ---------------------------------------------------------------------


def transform_weather(bronze: DataFrame) -> DataFrame:
    df = explode_payload(bronze, "weather").select(
        *SESSION_KEY,
        session_seconds("date").alias("session_time_s"),
        F.col("air_temperature").alias("air_temp_c"),
        F.col("track_temperature").alias("track_temp_c"),
        F.col("humidity").alias("humidity_pct"),
        F.col("pressure").alias("pressure_mbar"),
        (F.col("rainfall") > 0).alias("is_raining"),
        F.col("wind_speed").alias("wind_speed_ms"),
        F.col("wind_direction").alias("wind_direction_deg"),
        "_ingested_at",
        "_source_batch_id",
    )
    return latest_per_key(df, [*SESSION_KEY, "session_time_s"]).withColumn(
        "_updated_at", F.current_timestamp()
    )


def transform_car_data(bronze: DataFrame) -> DataFrame:
    df = explode_payload(bronze, "car_data").select(
        *SESSION_KEY,
        "driver_number",
        session_seconds("date").alias("session_time_s"),
        F.to_timestamp("date").alias("sample_utc"),
        F.col("speed").alias("speed_kmh"),
        "rpm",
        F.col("n_gear").alias("gear"),
        F.col("throttle").alias("throttle_pct"),
        (F.col("brake") > 0).alias("is_braking"),
        "drs",
        F.lit("openf1").alias("source"),
        "_ingested_at",
        "_source_batch_id",
    )
    keys = [*SESSION_KEY, "driver_number", "session_time_s"]
    return df.dropDuplicates(keys).withColumn("_updated_at", F.current_timestamp())


def transform_pos_data(bronze: DataFrame) -> DataFrame:
    df = explode_payload(bronze, "location").select(
        *SESSION_KEY,
        "driver_number",
        session_seconds("date").alias("session_time_s"),
        F.to_timestamp("date").alias("sample_utc"),
        # Same feed as FastF1: positions are in 1/10 m.
        (F.col("x") / 10).alias("x_m"),
        (F.col("y") / 10).alias("y_m"),
        (F.col("z") / 10).alias("z_m"),
        F.lit(None).cast("string").alias("status"),
        F.lit("openf1").alias("source"),
        "_ingested_at",
        "_source_batch_id",
    )
    keys = [*SESSION_KEY, "driver_number", "session_time_s"]
    return df.dropDuplicates(keys).withColumn("_updated_at", F.current_timestamp())
