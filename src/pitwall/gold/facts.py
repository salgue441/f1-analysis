"""Gold facts (docs/data-model.md#facts)."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pitwall.gold.keys import surrogate_key


def build_fact_session_result(results: DataFrame, qualifying: DataFrame) -> DataFrame:
    """Grain: driver x session. Race and sprint come from silver.results, qualifying from
    silver.qualifying (finish position = qualifying position, no points)."""
    race_like = results.select(
        "season",
        "round",
        "session_type",
        "driver_id",
        "constructor_id",
        "car_number",
        "grid",
        "finish_position",
        "position_text",
        "is_classified",
        "points",
        "laps_completed",
        "status",
        "race_time_s",
        F.col("fastest_lap_time_s").alias("best_lap_time_s"),
        "_source_batch_id",
    )
    quali = qualifying.select(
        "season",
        "round",
        F.lit("Q").alias("session_type"),
        "driver_id",
        "constructor_id",
        "car_number",
        F.lit(None).cast("int").alias("grid"),
        F.col("position").alias("finish_position"),
        F.col("position").cast("string").alias("position_text"),
        F.lit(True).alias("is_classified"),
        F.lit(0.0).alias("points"),
        F.lit(None).cast("int").alias("laps_completed"),
        F.lit(None).cast("string").alias("status"),
        F.lit(None).cast("double").alias("race_time_s"),
        F.col("best_time_s").alias("best_lap_time_s"),
        "_source_batch_id",
    )
    return (
        race_like.unionByName(quali)
        .select(
            surrogate_key("season", "round", "session_type", "driver_id").alias(
                "session_result_key"
            ),
            surrogate_key("season", "round", "session_type").alias("session_key"),
            surrogate_key("driver_id").alias("driver_key"),
            surrogate_key("constructor_id").alias("constructor_key"),
            "*",
        )
        .withColumn("_updated_at", F.current_timestamp())
    )
