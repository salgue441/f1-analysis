"""bronze.fastf1_* (except laps) -> silver: weather, track status, neutralization periods
and car telemetry."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from pitwall.quality import Expectation
from pitwall.silver.common import latest_per_key

SESSION_KEY = ["season", "event_round", "session_type"]

# FastF1 track status codes.
TRACK_STATUS_LABELS = {
    "1": "green",
    "2": "yellow",
    "4": "safety_car",
    "5": "red_flag",
    "6": "vsc",
    "7": "vsc_ending",
}
NEUTRALIZED_STATUSES = ("4", "5", "6", "7")


def _nan_to_null(col: str) -> Column:
    c = F.col(col).cast("double")
    return F.when(F.isnan(c), F.lit(None)).otherwise(c)


def _session_cols() -> list[Column]:
    return [
        F.col("season").cast("int").alias("season"),
        F.col("event_round").cast("int").alias("event_round"),
        F.col("session_type"),
    ]


def _lineage() -> list[Column]:
    return [F.col("_ingested_at"), F.col("_batch_id").alias("_source_batch_id")]


def latest_batch_per_session(df: DataFrame) -> DataFrame:
    """Keep only rows from the most recent ingestion of each session.

    Cheaper than a row_number dedup for telemetry: a small aggregate + broadcast join instead
    of a shuffle over every sample.
    """
    latest = df.groupBy(*SESSION_KEY).agg(F.max("_ingested_at").alias("_latest"))
    return (
        df.join(F.broadcast(latest), SESSION_KEY)
        .filter(F.col("_ingested_at") == F.col("_latest"))
        .drop("_latest")
    )


# --- weather ---------------------------------------------------------------------------------

WEATHER_KEY = [*SESSION_KEY, "session_time_s"]
WEATHER_EXPECTATIONS = [
    Expectation("natural_key_not_null", " AND ".join(f"{k} IS NOT NULL" for k in WEATHER_KEY)),
    Expectation("track_temp_plausible", "track_temp_c IS NULL OR track_temp_c BETWEEN -5 AND 75"),
    Expectation("air_temp_plausible", "air_temp_c IS NULL OR air_temp_c BETWEEN -10 AND 55"),
]


def transform_weather(bronze: DataFrame) -> DataFrame:
    df = bronze.select(
        *_session_cols(),
        _nan_to_null("time").alias("session_time_s"),
        _nan_to_null("air_temp").alias("air_temp_c"),
        _nan_to_null("track_temp").alias("track_temp_c"),
        _nan_to_null("humidity").alias("humidity_pct"),
        _nan_to_null("pressure").alias("pressure_mbar"),
        F.col("rainfall").cast("boolean").alias("is_raining"),
        _nan_to_null("wind_speed").alias("wind_speed_ms"),
        _nan_to_null("wind_direction").alias("wind_direction_deg"),
        *_lineage(),
    )
    return latest_per_key(df, WEATHER_KEY).withColumn("_updated_at", F.current_timestamp())


# --- track status -> neutralization periods ----------------------------------------------------

RACE_CONTROL_KEY = [*SESSION_KEY, "start_s"]
RACE_CONTROL_EXPECTATIONS = [
    Expectation("natural_key_not_null", " AND ".join(f"{k} IS NOT NULL" for k in RACE_CONTROL_KEY)),
    Expectation("status_known", "status_label IS NOT NULL", severity="warn"),
    Expectation("period_ordered", "end_s IS NULL OR end_s >= start_s"),
]


def transform_race_control(bronze: DataFrame) -> DataFrame:
    """One row per track status period: [start_s, end_s) in session time.

    Each FastF1 track status row marks a change; the period lasts until the next change.
    The last period of a session has `end_s` NULL (lasts until the session ends).
    """
    labels = F.create_map(*[F.lit(x) for kv in TRACK_STATUS_LABELS.items() for x in kv])
    changes = latest_per_key(
        bronze.select(
            *_session_cols(),
            _nan_to_null("time").alias("start_s"),
            F.col("status").cast("string").alias("status"),
            *_lineage(),
        ),
        RACE_CONTROL_KEY,
    )
    in_session = Window.partitionBy(*SESSION_KEY).orderBy("start_s")
    return (
        changes.withColumn("end_s", F.lead("start_s").over(in_session))
        .withColumn("status_label", labels[F.col("status")])
        .withColumn("is_neutralized", F.col("status").isin(*NEUTRALIZED_STATUSES))
        .withColumn("_updated_at", F.current_timestamp())
    )


# --- telemetry -------------------------------------------------------------------------------

TELEMETRY_KEY = [*SESSION_KEY, "driver_number", "session_time_s"]
CAR_EXPECTATIONS = [
    Expectation("natural_key_not_null", " AND ".join(f"{k} IS NOT NULL" for k in TELEMETRY_KEY)),
    Expectation("speed_plausible", "speed_kmh BETWEEN 0 AND 400"),
    Expectation("rpm_plausible", "rpm BETWEEN 0 AND 16000"),
    Expectation("gear_valid", "gear BETWEEN 0 AND 8"),
    # FastF1 throttle occasionally reads 104 (sensor artefact): kept, counted.
    Expectation("throttle_in_range", "throttle_pct BETWEEN 0 AND 100", severity="warn"),
]
POS_EXPECTATIONS = [
    Expectation("natural_key_not_null", " AND ".join(f"{k} IS NOT NULL" for k in TELEMETRY_KEY)),
    # OpenF1 has no position status; FastF1 does.
    Expectation("on_track", "status IS NULL OR status = 'OnTrack'", severity="warn"),
]


def _telemetry_keys() -> list[Column]:
    return [
        *_session_cols(),
        F.col("driver_number").cast("int").alias("driver_number"),
        _nan_to_null("session_time").alias("session_time_s"),
        F.col("date").cast("timestamp").alias("sample_utc"),
    ]


def transform_car_data(bronze: DataFrame) -> DataFrame:
    df = latest_batch_per_session(bronze).select(
        *_telemetry_keys(),
        F.col("speed").cast("double").alias("speed_kmh"),
        F.col("rpm").cast("double").alias("rpm"),
        F.col("n_gear").cast("int").alias("gear"),
        F.col("throttle").cast("double").alias("throttle_pct"),
        F.col("brake").cast("boolean").alias("is_braking"),
        F.col("drs").cast("int").alias("drs"),
        F.col("source").cast("string").alias("source"),
        *_lineage(),
    )
    # One batch per session after the filter above; drop exact repeats from FastF1 itself.
    return df.dropDuplicates(TELEMETRY_KEY).withColumn("_updated_at", F.current_timestamp())


def transform_pos_data(bronze: DataFrame) -> DataFrame:
    df = latest_batch_per_session(bronze).select(
        *_telemetry_keys(),
        # FastF1 positions are in 1/10 m.
        (F.col("x").cast("double") / 10).alias("x_m"),
        (F.col("y").cast("double") / 10).alias("y_m"),
        (F.col("z").cast("double") / 10).alias("z_m"),
        F.col("status").cast("string").alias("status"),
        F.col("source").cast("string").alias("source"),
        *_lineage(),
    )
    return df.dropDuplicates(TELEMETRY_KEY).withColumn("_updated_at", F.current_timestamp())
