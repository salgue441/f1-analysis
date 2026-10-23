"""Gold dimensions."""

from __future__ import annotations

import datetime as dt

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from pitwall.gold.keys import surrogate_key

OPEN_END_DATE = dt.date(9999, 12, 31)


def build_dim_driver_team(entries: DataFrame) -> DataFrame:
    """SCD2 driver -> constructor spells from race-by-race entries.

    `entries` needs: driver_id, constructor_id, race_date (date), round (int).
    A new spell starts whenever a driver's constructor differs from their previous race,
    so mid-season swaps and returns to a former team both create new rows.
    `is_current` marks each driver's latest known spell.
    """
    by_race = Window.partitionBy("driver_id").orderBy("race_date", "round")
    starts = (
        entries.select("driver_id", "constructor_id", "race_date", "round")
        .dropDuplicates(["driver_id", "race_date"])
        .withColumn("_prev_constructor", F.lag("constructor_id").over(by_race))
        .filter(
            F.col("_prev_constructor").isNull()
            | (F.col("_prev_constructor") != F.col("constructor_id"))
        )
    )
    by_start = Window.partitionBy("driver_id").orderBy("race_date")
    return (
        starts.withColumn("_next_start", F.lead("race_date").over(by_start))
        .select(
            surrogate_key("driver_id", "constructor_id", "race_date").alias("driver_team_key"),
            surrogate_key("driver_id").alias("driver_key"),
            surrogate_key("constructor_id").alias("constructor_key"),
            "driver_id",
            "constructor_id",
            F.col("race_date").alias("valid_from"),
            F.coalesce(F.date_sub("_next_start", 1), F.lit(OPEN_END_DATE)).alias("valid_to"),
            F.col("_next_start").isNull().alias("is_current"),
        )
        .withColumn("_updated_at", F.current_timestamp())
    )
