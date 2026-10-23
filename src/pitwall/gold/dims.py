"""Gold dimensions (docs/data-model.md#dimensions)."""

from __future__ import annotations

import datetime as dt

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from pitwall.gold.keys import surrogate_key

OPEN_END_DATE = dt.date(9999, 12, 31)

COMPOUNDS = [
    # compound, is_dry, display order
    ("SOFT", True, 1),
    ("MEDIUM", True, 2),
    ("HARD", True, 3),
    ("INTERMEDIATE", False, 4),
    ("WET", False, 5),
    ("UNKNOWN", False, 6),
]

# Constructor ids that are the same team (same factory) under different names.
# Lets analyses follow a team across rebrands without merging the constructors themselves.
CONSTRUCTOR_LINEAGE = {
    "minardi": "faenza",
    "toro_rosso": "faenza",
    "alphatauri": "faenza",
    "rb": "faenza",
    "jordan": "silverstone",
    "midland": "silverstone",
    "spyker": "silverstone",
    "force_india": "silverstone",
    "racing_point": "silverstone",
    "aston_martin": "silverstone",
    "sauber": "hinwil",
    "bmw_sauber": "hinwil",
    "alfa": "hinwil",
    "toleman": "enstone",
    "benetton": "enstone",
    "renault": "enstone",
    "lotus_f1": "enstone",
    "alpine": "enstone",
    "stewart": "milton_keynes",
    "jaguar": "milton_keynes",
    "red_bull": "milton_keynes",
    "tyrrell": "brackley",
    "bar": "brackley",
    "honda": "brackley",
    "brawn": "brackley",
    "mercedes": "brackley",
}

# Session code -> column in silver.races holding its UTC start.
SESSION_START_COLUMNS = {
    "FP1": "fp1_start_utc",
    "FP2": "fp2_start_utc",
    "FP3": "fp3_start_utc",
    "Q": "q_start_utc",
    "SQ": "sq_start_utc",
    "SS": "ss_start_utc",
    "S": "s_start_utc",
    "R": "race_start_utc",
}


def _updated(df: DataFrame) -> DataFrame:
    return df.withColumn("_updated_at", F.current_timestamp())


def date_key(col: str) -> Column:
    """yyyymmdd as an int: readable and stable."""
    return F.date_format(F.col(col), "yyyyMMdd").cast("int")


def build_dim_driver(drivers: DataFrame) -> DataFrame:
    return _updated(
        drivers.select(
            surrogate_key("driver_id").alias("driver_key"),
            "driver_id",
            "code",
            "permanent_number",
            "given_name",
            "family_name",
            F.concat_ws(" ", "given_name", "family_name").alias("full_name"),
            "nationality",
            "date_of_birth",
            "_source_batch_id",
        )
    )


def build_dim_constructor(constructors: DataFrame) -> DataFrame:
    lineage = F.create_map(*[F.lit(x) for kv in CONSTRUCTOR_LINEAGE.items() for x in kv])
    return _updated(
        constructors.select(
            surrogate_key("constructor_id").alias("constructor_key"),
            "constructor_id",
            "constructor_name",
            "nationality",
            F.coalesce(lineage[F.col("constructor_id")], F.col("constructor_id")).alias(
                "lineage_group"
            ),
            "_source_batch_id",
        )
    )


def build_dim_circuit(circuits: DataFrame) -> DataFrame:
    """Static circuit attributes. Derived attributes (pit loss, degradation tier) live in
    feature tables, so gold stays a function of the sources only."""
    return _updated(
        circuits.select(
            surrogate_key("circuit_id").alias("circuit_key"),
            "circuit_id",
            "circuit_name",
            "locality",
            "country",
            "lat",
            "lon",
            "_source_batch_id",
        )
    )


def build_dim_compound(spark: SparkSession) -> DataFrame:
    df = spark.createDataFrame(COMPOUNDS, "compound string, is_dry boolean, sort_order int")
    return _updated(df.select(surrogate_key("compound").alias("compound_key"), "*"))


def build_dim_date(spark: SparkSession, start: dt.date, end: dt.date) -> DataFrame:
    days = spark.sql(
        f"SELECT explode(sequence(DATE'{start.isoformat()}', DATE'{end.isoformat()}')) AS date"
    )
    return _updated(
        days.select(
            date_key("date").alias("date_key"),
            "date",
            F.year("date").alias("year"),
            F.quarter("date").alias("quarter"),
            F.month("date").alias("month"),
            F.dayofmonth("date").alias("day"),
            F.date_format("date", "EEEE").alias("day_name"),
            # ISO day of week: Monday = 1 ... Sunday = 7.
            ((F.dayofweek("date") + 5) % 7 + 1).alias("iso_day_of_week"),
            F.weekofyear("date").alias("iso_week"),
            F.dayofweek("date").isin(1, 7).alias("is_weekend"),
        )
    )


def build_dim_session(
    races: DataFrame, results: DataFrame, observed: DataFrame | None = None
) -> DataFrame:
    """One row per session: every session with a scheduled start, plus any session seen in
    `observed` (season, round, session_type). Older seasons have no practice/qualifying
    times in Jolpica, but their qualifying results still need a session to point at.

    `scheduled_laps` comes from the winner's completed laps (race and sprint only). For a
    race shortened by a red flag that is the distance actually run, which is what the
    fuel proxy needs.
    """
    keys = ["season", "round", "session_type"]
    starts = races.select(
        "season",
        "round",
        F.explode(
            F.array(
                *[
                    F.struct(F.lit(code).alias("session_type"), F.col(col).alias("start_utc"))
                    for code, col in SESSION_START_COLUMNS.items()
                ]
            )
        ).alias("s"),
    ).select("season", "round", "s.session_type", "s.start_utc")
    session_ids = starts.filter(F.col("start_utc").isNotNull()).select(keys)
    if observed is not None:
        session_ids = session_ids.unionByName(observed.select(keys))

    laps = results.filter(F.col("finish_position") == 1).select(
        *keys, F.col("laps_completed").alias("scheduled_laps")
    )
    return _updated(
        session_ids.distinct()
        .join(starts, keys, "left")
        .join(
            races.select(
                "season", "round", "race_name", "circuit_id", "race_date", "_source_batch_id"
            ),
            ["season", "round"],
        )
        .join(laps, keys, "left")
        .select(
            surrogate_key(*keys).alias("session_key"),
            surrogate_key("circuit_id").alias("circuit_key"),
            # Sessions without a known start fall back to the race date.
            F.date_format(F.coalesce(F.to_date("start_utc"), F.col("race_date")), "yyyyMMdd")
            .cast("int")
            .alias("date_key"),
            *keys,
            F.col("race_name").alias("event_name"),
            "circuit_id",
            F.col("start_utc").alias("start_time_utc"),
            "scheduled_laps",
            F.when(F.col("season") >= 2022, "ground_effect").otherwise("pre_2022").alias("era"),
            "_source_batch_id",
        )
    )


def driver_team_entries(results: DataFrame, races: DataFrame) -> DataFrame:
    """Race entries (driver, constructor, race date) feeding the SCD2 dimension."""
    return results.filter(F.col("session_type") == "R").join(
        races.select("season", "round", "race_date"), ["season", "round"]
    )


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
