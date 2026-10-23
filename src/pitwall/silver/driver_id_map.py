"""silver.driver_id_map: Jolpica driver_id <-> car number <-> code, per race weekend.

FastF1 identifies drivers by car number and three-letter code; Jolpica by `driver_id`. Car
numbers are reused across seasons (and occasionally change within one), so the mapping is
built per (season, round) from Jolpica's own entry lists, which carry all three.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

KEY = ["season", "round", "driver_number"]


class DriverMapError(RuntimeError):
    pass


def build_driver_id_map(results: DataFrame, qualifying: DataFrame, drivers: DataFrame) -> DataFrame:
    """One row per (season, round, driver_number). Race entries win over sprint/qualifying."""
    priority = {"R": 0, "S": 1, "Q": 2}
    entries = results.select(
        "season", "round", "session_type", "driver_id", "constructor_id", "car_number"
    ).unionByName(
        qualifying.select(
            "season",
            "round",
            F.lit("Q").alias("session_type"),
            "driver_id",
            "constructor_id",
            "car_number",
        )
    )
    rank = F.create_map(*[F.lit(x) for kv in priority.items() for x in kv])
    best = (
        entries.filter(F.col("car_number").isNotNull())
        .withColumn("_priority", rank[F.col("session_type")])
        .groupBy("season", "round", "driver_id")
        .agg(F.min_by(F.struct("car_number", "constructor_id"), "_priority").alias("e"))
    )
    return (
        best.select(
            "season",
            "round",
            F.col("e.car_number").alias("driver_number"),
            "driver_id",
            F.col("e.constructor_id").alias("constructor_id"),
        )
        .join(drivers.select("driver_id", "code"), "driver_id", "left")
        .withColumn("_updated_at", F.current_timestamp())
    )


def check_one_to_one(mapping: DataFrame) -> None:
    """Within a race weekend, a car number maps to one driver and a driver to one number."""
    for key, other in (("driver_number", "driver_id"), ("driver_id", "driver_number")):
        clashes = (
            mapping.groupBy("season", "round", key)
            .agg(F.countDistinct(other).alias("n"))
            .filter(F.col("n") > 1)
            .limit(5)
            .collect()
        )
        if clashes:
            raise DriverMapError(f"{key} -> {other} is not 1:1: {[r.asDict() for r in clashes]}")
