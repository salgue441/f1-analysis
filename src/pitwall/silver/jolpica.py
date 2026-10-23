"""bronze.jolpica_* -> silver entity tables.

Bronze stores each API page as a raw JSON string. Parsing happens here against one explicit
schema, so a source change shows up as NULLs that the expectations catch instead of columns
silently shifting. Jolpica serves every value as a string; typing happens here too.

Pagination splits a race's results across pages, so a race can appear in several pages with
partial `Results`. Exploding then deduplicating on the natural key handles that.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from pitwall.quality import Expectation
from pitwall.silver.common import duration_s, latest_per_key, utc_timestamp

_DRIVER = (
    "struct<driverId:string, permanentNumber:string, code:string, givenName:string, "
    "familyName:string, dateOfBirth:string, nationality:string>"
)
_CONSTRUCTOR = "struct<constructorId:string, name:string, nationality:string>"
_CIRCUIT = (
    "struct<circuitId:string, circuitName:string, "
    "Location:struct<lat:string, long:string, locality:string, country:string>>"
)
_SESSION = "struct<date:string, time:string>"
_RESULT = (
    f"struct<number:string, position:string, positionText:string, points:string, "
    f"Driver:{_DRIVER}, Constructor:{_CONSTRUCTOR}, grid:string, laps:string, status:string, "
    f"Time:struct<millis:string, time:string>, "
    f"FastestLap:struct<rank:string, lap:string, Time:struct<time:string>>>"
)
_QUALIFYING = (
    f"struct<number:string, position:string, Driver:{_DRIVER}, Constructor:{_CONSTRUCTOR}, "
    f"Q1:string, Q2:string, Q3:string>"
)
_PIT_STOP = "struct<driverId:string, lap:string, stop:string, time:string, duration:string>"
_RACE = (
    f"struct<season:string, round:string, raceName:string, Circuit:{_CIRCUIT}, "
    f"date:string, time:string, FirstPractice:{_SESSION}, SecondPractice:{_SESSION}, "
    f"ThirdPractice:{_SESSION}, Qualifying:{_SESSION}, Sprint:{_SESSION}, "
    f"SprintQualifying:{_SESSION}, SprintShootout:{_SESSION}, "
    f"Results:array<{_RESULT}>, SprintResults:array<{_RESULT}>, "
    f"QualifyingResults:array<{_QUALIFYING}>, PitStops:array<{_PIT_STOP}>>"
)
PAYLOAD_SCHEMA = (
    f"struct<MRData:struct<total:string, "
    f"RaceTable:struct<Races:array<{_RACE}>>, "
    f"DriverTable:struct<Drivers:array<{_DRIVER}>>, "
    f"ConstructorTable:struct<Constructors:array<{_CONSTRUCTOR}>>, "
    f"CircuitTable:struct<Circuits:array<{_CIRCUIT}>>>>"
)

# Session code (FastF1 naming) -> field on a Jolpica race.
SESSION_FIELDS = {
    "FP1": "FirstPractice",
    "FP2": "SecondPractice",
    "FP3": "ThirdPractice",
    "Q": "Qualifying",
    "SQ": "SprintQualifying",
    "SS": "SprintShootout",
    "S": "Sprint",
}


def _lineage() -> list[Column]:
    return [F.col("_ingested_at"), F.col("_batch_id").alias("_source_batch_id")]


def _parse(bronze: DataFrame) -> DataFrame:
    return bronze.select(F.from_json("payload", PAYLOAD_SCHEMA)["MRData"].alias("mr"), *_lineage())


def _races(bronze: DataFrame) -> DataFrame:
    return _parse(bronze).select(
        F.explode("mr.RaceTable.Races").alias("race"), "_ingested_at", "_source_batch_id"
    )


def _finish(df: DataFrame, keys: list[str]) -> DataFrame:
    return latest_per_key(df, keys).withColumn("_updated_at", F.current_timestamp())


def _int(col: str | Column) -> Column:
    return (F.col(col) if isinstance(col, str) else col).cast("int")


# --- races ----------------------------------------------------------------------------------

RACES_KEY = ["season", "round"]
RACES_EXPECTATIONS = [
    Expectation("natural_key_not_null", "season IS NOT NULL AND round IS NOT NULL"),
    Expectation("circuit_present", "circuit_id IS NOT NULL"),
    Expectation("race_date_present", "race_date IS NOT NULL"),
]


def transform_races(bronze: DataFrame) -> DataFrame:
    """One row per race weekend, with UTC start times of every scheduled session."""
    race = F.col("race")
    session_starts = [
        utc_timestamp(race[field]["date"], race[field]["time"]).alias(f"{code.lower()}_start_utc")
        for code, field in SESSION_FIELDS.items()
    ]
    df = _races(bronze).select(
        _int(race["season"]).alias("season"),
        _int(race["round"]).alias("round"),
        race["raceName"].alias("race_name"),
        race["Circuit"]["circuitId"].alias("circuit_id"),
        F.to_date(race["date"]).alias("race_date"),
        utc_timestamp(race["date"], race["time"]).alias("race_start_utc"),
        *session_starts,
        "_ingested_at",
        "_source_batch_id",
    )
    return _finish(df, RACES_KEY)


# --- circuits, drivers, constructors (season-independent entities) ---------------------------

CIRCUITS_KEY = ["circuit_id"]
CIRCUITS_EXPECTATIONS = [
    Expectation("natural_key_not_null", "circuit_id IS NOT NULL"),
    Expectation("coordinates_valid", "lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180"),
]


def transform_circuits(bronze: DataFrame) -> DataFrame:
    c = F.col("c")
    df = _parse(bronze).select(
        F.explode("mr.CircuitTable.Circuits").alias("c"), "_ingested_at", "_source_batch_id"
    )
    df = df.select(
        c["circuitId"].alias("circuit_id"),
        c["circuitName"].alias("circuit_name"),
        c["Location"]["locality"].alias("locality"),
        c["Location"]["country"].alias("country"),
        c["Location"]["lat"].cast("double").alias("lat"),
        c["Location"]["long"].cast("double").alias("lon"),
        "_ingested_at",
        "_source_batch_id",
    )
    return _finish(df, CIRCUITS_KEY)


DRIVERS_KEY = ["driver_id"]
DRIVERS_EXPECTATIONS = [
    Expectation("natural_key_not_null", "driver_id IS NOT NULL"),
    Expectation("name_present", "family_name IS NOT NULL"),
    Expectation("code_present", "code IS NOT NULL", severity="warn"),
]


def transform_drivers(bronze: DataFrame) -> DataFrame:
    d = F.col("d")
    df = _parse(bronze).select(
        F.explode("mr.DriverTable.Drivers").alias("d"), "_ingested_at", "_source_batch_id"
    )
    df = df.select(
        d["driverId"].alias("driver_id"),
        d["code"].alias("code"),
        _int(d["permanentNumber"]).alias("permanent_number"),
        d["givenName"].alias("given_name"),
        d["familyName"].alias("family_name"),
        F.to_date(d["dateOfBirth"]).alias("date_of_birth"),
        d["nationality"].alias("nationality"),
        "_ingested_at",
        "_source_batch_id",
    )
    return _finish(df, DRIVERS_KEY)


CONSTRUCTORS_KEY = ["constructor_id"]
CONSTRUCTORS_EXPECTATIONS = [
    Expectation("natural_key_not_null", "constructor_id IS NOT NULL"),
    Expectation("name_present", "constructor_name IS NOT NULL"),
]


def transform_constructors(bronze: DataFrame) -> DataFrame:
    c = F.col("c")
    df = _parse(bronze).select(
        F.explode("mr.ConstructorTable.Constructors").alias("c"),
        "_ingested_at",
        "_source_batch_id",
    )
    df = df.select(
        c["constructorId"].alias("constructor_id"),
        c["name"].alias("constructor_name"),
        c["nationality"].alias("nationality"),
        "_ingested_at",
        "_source_batch_id",
    )
    return _finish(df, CONSTRUCTORS_KEY)


# --- results (race + sprint) -----------------------------------------------------------------

RESULTS_KEY = ["season", "round", "session_type", "driver_id"]
RESULTS_EXPECTATIONS = [
    Expectation("natural_key_not_null", " AND ".join(f"{k} IS NOT NULL" for k in RESULTS_KEY)),
    Expectation("constructor_present", "constructor_id IS NOT NULL"),
    Expectation("finish_position_valid", "finish_position BETWEEN 1 AND 30"),
    Expectation("points_non_negative", "points >= 0"),
    Expectation("grid_valid", "grid IS NULL OR grid BETWEEN 0 AND 30"),
    Expectation("car_number_present", "car_number IS NOT NULL", severity="warn"),
]


def _results(bronze: DataFrame, field: str, session_type: str) -> DataFrame:
    r = F.col("r")
    df = _races(bronze).select(
        F.col("race.season").alias("season"),
        F.col("race.round").alias("round"),
        F.explode(F.col("race")[field]).alias("r"),
        "_ingested_at",
        "_source_batch_id",
    )
    return df.select(
        _int("season").alias("season"),
        _int("round").alias("round"),
        F.lit(session_type).alias("session_type"),
        r["Driver"]["driverId"].alias("driver_id"),
        r["Constructor"]["constructorId"].alias("constructor_id"),
        _int(r["number"]).alias("car_number"),
        # grid 0 means pit-lane start in Jolpica.
        _int(r["grid"]).alias("grid"),
        _int(r["position"]).alias("finish_position"),
        r["positionText"].alias("position_text"),
        r["positionText"].rlike("^[0-9]+$").alias("is_classified"),
        r["points"].cast("double").alias("points"),
        _int(r["laps"]).alias("laps_completed"),
        r["status"].alias("status"),
        (r["Time"]["millis"].cast("double") / 1000).alias("race_time_s"),
        _int(r["FastestLap"]["rank"]).alias("fastest_lap_rank"),
        _int(r["FastestLap"]["lap"]).alias("fastest_lap_number"),
        duration_s(r["FastestLap"]["Time"]["time"]).alias("fastest_lap_time_s"),
        "_ingested_at",
        "_source_batch_id",
    )


def transform_results(bronze: DataFrame) -> DataFrame:
    """bronze.jolpica_results -> silver.results (session_type 'R')."""
    return _finish(_results(bronze, "Results", "R"), RESULTS_KEY)


def transform_sprint_results(bronze: DataFrame) -> DataFrame:
    """bronze.jolpica_sprint -> silver.results (session_type 'S'); merged into the same table."""
    return _finish(_results(bronze, "SprintResults", "S"), RESULTS_KEY)


# --- qualifying ------------------------------------------------------------------------------

QUALIFYING_KEY = ["season", "round", "driver_id"]
QUALIFYING_EXPECTATIONS = [
    Expectation("natural_key_not_null", " AND ".join(f"{k} IS NOT NULL" for k in QUALIFYING_KEY)),
    Expectation("position_valid", "position BETWEEN 1 AND 30"),
    Expectation("q_time_plausible", "best_time_s IS NULL OR best_time_s BETWEEN 50 AND 200"),
    Expectation("has_a_time", "best_time_s IS NOT NULL", severity="warn"),
]


def transform_qualifying(bronze: DataFrame) -> DataFrame:
    q = F.col("q")
    df = _races(bronze).select(
        F.col("race.season").alias("season"),
        F.col("race.round").alias("round"),
        F.explode("race.QualifyingResults").alias("q"),
        "_ingested_at",
        "_source_batch_id",
    )
    df = df.select(
        _int("season").alias("season"),
        _int("round").alias("round"),
        q["Driver"]["driverId"].alias("driver_id"),
        q["Constructor"]["constructorId"].alias("constructor_id"),
        _int(q["number"]).alias("car_number"),
        _int(q["position"]).alias("position"),
        duration_s(q["Q1"]).alias("q1_s"),
        duration_s(q["Q2"]).alias("q2_s"),
        duration_s(q["Q3"]).alias("q3_s"),
        "_ingested_at",
        "_source_batch_id",
    ).withColumn("best_time_s", F.least("q1_s", "q2_s", "q3_s"))
    return _finish(df, QUALIFYING_KEY)


# --- pit stops -------------------------------------------------------------------------------

PIT_STOPS_KEY = ["season", "round", "driver_id", "stop_number"]
PIT_STOPS_EXPECTATIONS = [
    Expectation("natural_key_not_null", " AND ".join(f"{k} IS NOT NULL" for k in PIT_STOPS_KEY)),
    Expectation("lap_positive", "lap >= 1"),
    # Red-flag "stops" can last many minutes; they are kept but counted.
    Expectation("duration_typical", "duration_s BETWEEN 10 AND 60", severity="warn"),
]


def transform_pit_stops(bronze: DataFrame) -> DataFrame:
    """Pit stops as reported by Jolpica. `duration_s` is pit-lane entry to exit."""
    p = F.col("p")
    df = _races(bronze).select(
        F.col("race.season").alias("season"),
        F.col("race.round").alias("round"),
        F.explode("race.PitStops").alias("p"),
        "_ingested_at",
        "_source_batch_id",
    )
    df = df.select(
        _int("season").alias("season"),
        _int("round").alias("round"),
        p["driverId"].alias("driver_id"),
        _int(p["stop"]).alias("stop_number"),
        _int(p["lap"]).alias("lap"),
        duration_s(p["duration"]).alias("duration_s"),
        "_ingested_at",
        "_source_batch_id",
    )
    return _finish(df, PIT_STOPS_KEY)
