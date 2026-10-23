import datetime as dt

import pytest

from pitwall.gold import dims, facts
from pitwall.gold.integrity import IntegrityError, check_foreign_keys
from pitwall.gold.keys import surrogate_key
from pitwall.gold.writer import overwrite_season
from pitwall.silver.merge import merge_into_delta

RACES_SCHEMA = (
    "season int, round int, race_name string, circuit_id string, race_date date, "
    "race_start_utc timestamp, fp1_start_utc timestamp, fp2_start_utc timestamp, "
    "fp3_start_utc timestamp, q_start_utc timestamp, sq_start_utc timestamp, "
    "ss_start_utc timestamp, s_start_utc timestamp, _source_batch_id string"
)
RESULTS_SCHEMA = (
    "season int, round int, session_type string, driver_id string, constructor_id string, "
    "car_number int, grid int, finish_position int, position_text string, "
    "is_classified boolean, points double, laps_completed int, status string, "
    "race_time_s double, fastest_lap_time_s double, _source_batch_id string"
)
QUALI_SCHEMA = (
    "season int, round int, driver_id string, constructor_id string, car_number int, "
    "position int, best_time_s double, _source_batch_id string"
)


def _races(spark):
    ts = dt.datetime
    return spark.createDataFrame(
        [
            (2024, 1, "Bahrain GP", "bahrain", dt.date(2024, 3, 2), ts(2024, 3, 2, 15))
            + (None,) * 3
            + (ts(2024, 3, 1, 16), None, None, None, "b1"),
            # Older season: no session times except the race.
            (2018, 1, "Australian GP", "albert_park", dt.date(2018, 3, 25), ts(2018, 3, 25, 5))
            + (None,) * 7
            + ("b1",),
        ],
        RACES_SCHEMA,
    )


def _results(spark):
    return spark.createDataFrame(
        [
            (
                2024,
                1,
                "R",
                "max_verstappen",
                "red_bull",
                1,
                1,
                1,
                "1",
                True,
                26.0,
                57,
                "Finished",
                5504.7,
                92.6,
                "b1",
            ),
            (
                2024,
                1,
                "R",
                "perez",
                "red_bull",
                11,
                5,
                2,
                "2",
                True,
                18.0,
                57,
                "Finished",
                5527.0,
                93.0,
                "b1",
            ),
            (
                2018,
                1,
                "R",
                "vettel",
                "ferrari",
                5,
                3,
                1,
                "1",
                True,
                25.0,
                58,
                "Finished",
                5236.0,
                86.0,
                "b1",
            ),
        ],
        RESULTS_SCHEMA,
    )


def test_dim_session_covers_scheduled_and_observed_sessions(spark):
    quali = spark.createDataFrame([(2018, 1, "Q")], "season int, round int, session_type string")
    rows = {
        (r.season, r.session_type): r
        for r in dims.build_dim_session(_races(spark), _results(spark), quali).collect()
    }
    assert set(rows) == {(2024, "Q"), (2024, "R"), (2018, "R"), (2018, "Q")}
    assert rows[(2024, "R")].scheduled_laps == 57
    assert rows[(2024, "R")].era == "ground_effect" and rows[(2018, "R")].era == "pre_2022"
    assert rows[(2024, "Q")].date_key == 20240301
    # Unknown start: date falls back to race day.
    assert rows[(2018, "Q")].start_time_utc is None and rows[(2018, "Q")].date_key == 20180325


def test_dim_date_and_compound(spark):
    days = dims.build_dim_date(spark, dt.date(2024, 3, 1), dt.date(2024, 3, 3)).collect()
    assert [d.date_key for d in days] == [20240301, 20240302, 20240303]
    assert [d.iso_day_of_week for d in days] == [5, 6, 7]
    assert [d.is_weekend for d in days] == [False, True, True]

    compounds = {r.compound: r.is_dry for r in dims.build_dim_compound(spark).collect()}
    assert compounds["SOFT"] and not compounds["WET"]


def test_constructor_lineage(spark):
    constructors = spark.createDataFrame(
        [
            ("alphatauri", "AlphaTauri", "Italian", "b"),
            ("rb", "RB", "Italian", "b"),
            ("haas", "Haas", "American", "b"),
        ],
        "constructor_id string, constructor_name string, nationality string, "
        "_source_batch_id string",
    )
    groups = {
        r.constructor_id: r.lineage_group
        for r in dims.build_dim_constructor(constructors).collect()
    }
    assert groups == {"alphatauri": "faenza", "rb": "faenza", "haas": "haas"}


def test_fact_session_result_and_foreign_keys(spark):
    quali = spark.createDataFrame(
        [(2024, 1, "max_verstappen", "red_bull", 1, 1, 89.2, "b1")], QUALI_SCHEMA
    )
    fact = facts.build_fact_session_result(_results(spark), quali)
    assert fact.count() == 4
    q = fact.filter("session_type = 'Q'").first()
    assert q.finish_position == 1 and q.points == 0.0 and q.best_lap_time_s == 89.2

    observed = fact.select("season", "round", "session_type")
    dim_session = dims.build_dim_session(_races(spark), _results(spark), observed)
    drivers = spark.createDataFrame(
        [("max_verstappen",), ("perez",), ("vettel",)], "driver_id string"
    ).select(surrogate_key("driver_id").alias("driver_key"))
    check_foreign_keys(fact, {"session_key": dim_session, "driver_key": drivers})

    with pytest.raises(IntegrityError):
        check_foreign_keys(fact, {"driver_key": drivers.limit(1)})


def test_merge_and_season_overwrite_are_idempotent(spark, tmp_path):
    results = _results(spark)
    silver = str(tmp_path / "silver_results")
    for _ in range(2):
        merge_into_delta(spark, results, silver, ["season", "round", "session_type", "driver_id"])
    assert spark.read.format("delta").load(silver).count() == 3

    gold = str(tmp_path / "gold_fact")
    for _ in range(2):
        overwrite_season(spark, results, gold, 2024)
    overwrite_season(spark, results, gold, 2018)
    df = spark.read.format("delta").load(gold)
    assert df.filter("season = 2024").count() == 2 and df.filter("season = 2018").count() == 1
