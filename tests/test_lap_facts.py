import datetime as dt

import pytest
from pyspark.sql import functions as F

from pitwall.asof import asof_backward
from pitwall.gold import lap_facts
from pitwall.gold.integrity import check_unique
from pitwall.silver import fastf1
from pitwall.silver.driver_id_map import DriverMapError, build_driver_id_map, check_one_to_one

T0 = dt.datetime(2024, 3, 4)
LAPS_SCHEMA = (
    "season int, event_round int, session_type string, driver_number int, lap_number int, "
    "stint_number int, compound string, tyre_life_laps int, fresh_tyre boolean, "
    "lap_time_s double, sector1_s double, sector2_s double, sector3_s double, "
    "lap_start_time_s double, lap_end_time_s double, position_end_of_lap int, "
    "speed_trap_kmh double, is_pit_in_lap boolean, is_pit_out_lap boolean, "
    "is_green_flag boolean, track_status string, is_accurate boolean, is_deleted boolean, "
    "_source_batch_id string"
)
WEATHER_SCHEMA = (
    "season int, event_round int, session_type string, session_time_s double, "
    "track_temp_c double, air_temp_c double, is_raining boolean"
)
MAP_SCHEMA = "season int, round int, driver_number int, driver_id string, constructor_id string"


def _lap(driver, lap, start, time, pos, compound="SOFT", life=None, status="1", pit_in=False):
    return (
        2024,
        1,
        "R",
        driver,
        lap,
        1,
        compound,
        life or lap,
        False,
        time,
        30.0,
        30.0,
        time - 60.0,
        start,
        start + time,
        pos,
        300.0,
        pit_in,
        False,
        status == "1",
        status,
        True,
        False,
        "b1",
    )


@pytest.fixture
def laps(spark):
    return spark.createDataFrame(
        [
            _lap(1, 1, 100.0, 95.0, 1),
            _lap(11, 1, 100.0, 96.5, 2),
            _lap(1, 2, 195.0, 94.0, 1),
            _lap(11, 2, 196.5, 93.0, 2, status="4"),
        ],
        LAPS_SCHEMA,
    )


def _weather(spark, later_temp=40.0):
    return spark.createDataFrame(
        [
            (2024, 1, "R", 0.0, 30.0, 20.0, False),
            (2024, 1, "R", 150.0, 35.0, 21.0, False),
            (2024, 1, "R", 196.0, later_temp, 22.0, True),
        ],
        WEATHER_SCHEMA,
    )


def _map(spark):
    return spark.createDataFrame(
        [(2024, 1, 1, "max_verstappen", "red_bull"), (2024, 1, 11, "perez", "red_bull")],
        MAP_SCHEMA,
    )


def test_asof_backward_is_point_in_time(spark, laps):
    def temps(weather):
        joined = asof_backward(
            laps,
            weather,
            lap_facts.SESSION,
            "lap_start_time_s",
            "session_time_s",
            ["track_temp_c"],
        )
        return {(r.driver_number, r.lap_number): r.track_temp_c for r in joined.collect()}

    base = temps(_weather(spark))
    assert base == {(1, 1): 30.0, (11, 1): 30.0, (1, 2): 35.0, (11, 2): 40.0}
    # Changing a sample after a lap started must not change that lap.
    changed = temps(_weather(spark, later_temp=99.0))
    assert changed[(1, 2)] == 35.0 and changed[(11, 2)] == 99.0


def test_fact_lap_keys_gap_weather_and_flags(spark, laps):
    rows = {
        (r.driver_number, r.lap_number): r
        for r in lap_facts.build_fact_lap(laps, _weather(spark), _map(spark)).collect()
    }
    assert rows[(11, 1)].gap_to_car_ahead_s == pytest.approx(1.5)
    assert rows[(1, 1)].gap_to_car_ahead_s is None
    assert rows[(11, 2)].is_neutralized and not rows[(1, 2)].is_neutralized
    assert rows[(11, 2)].is_raining is True and rows[(1, 2)].is_raining is False
    assert rows[(1, 1)].driver_id == "max_verstappen"
    assert len({r.lap_key for r in rows.values()}) == 4


def test_fact_lap_unknown_compound_maps_to_unknown(spark):
    laps = spark.createDataFrame([_lap(1, 1, 0.0, 95.0, 1, compound="TEST_UNKNOWN")], LAPS_SCHEMA)
    row = lap_facts.build_fact_lap(laps, _weather(spark), _map(spark)).first()
    assert row.compound == "UNKNOWN"


def test_fact_stint_slope_uses_clean_laps(spark):
    laps = [_lap(1, n, 100.0 * n, 90.0 + 0.1 * n, 1, life=n) for n in range(1, 11)]
    laps.append(_lap(1, 11, 1100.0, 120.0, 1, life=11, pit_in=True))  # in-lap: excluded
    fact_lap = lap_facts.build_fact_lap(
        spark.createDataFrame(laps, LAPS_SCHEMA), _weather(spark), _map(spark)
    )
    stint = lap_facts.build_fact_stint(fact_lap).first()
    assert (stint.start_lap, stint.end_lap, stint.laps, stint.clean_laps) == (1, 11, 11, 10)
    assert stint.raw_deg_slope_s_per_lap == pytest.approx(0.1)
    assert stint.compound == "SOFT"


def test_fact_pit_stop_context(spark):
    laps = spark.createDataFrame(
        [
            _lap(1, 20, 2000.0, 110.0, 1, life=20, status="14", pit_in=True),
            _lap(1, 21, 2110.0, 115.0, 3, compound="HARD", life=1),
        ],
        LAPS_SCHEMA,
    )
    fact_lap = lap_facts.build_fact_lap(laps, _weather(spark), _map(spark))
    stops = spark.createDataFrame(
        [(2024, 1, "max_verstappen", 1, 20, 22.5, "b1")],
        "season int, round int, driver_id string, stop_number int, lap int, duration_s double, "
        "_source_batch_id string",
    )
    stop = lap_facts.build_fact_pit_stop(stops, fact_lap, _map(spark)).first()
    assert (stop.compound_before, stop.compound_after) == ("SOFT", "HARD")
    assert (stop.position_before, stop.position_after) == (1, 3)
    assert stop.under_neutralization is True and stop.driver_number == 1


def test_race_control_periods(spark):
    bronze = spark.createDataFrame(
        [
            (2024, 1, "R", 0.0, "1", T0, "b1"),
            (2024, 1, "R", 600.0, "4", T0, "b1"),
            (2024, 1, "R", 900.0, "1", T0, "b1"),
        ],
        "season int, event_round int, session_type string, time double, status string, "
        "_ingested_at timestamp, _batch_id string",
    )
    rows = fastf1.transform_race_control(bronze).orderBy("start_s").collect()
    assert [(r.start_s, r.end_s, r.status_label, r.is_neutralized) for r in rows] == [
        (0.0, 600.0, "green", False),
        (600.0, 900.0, "safety_car", True),
        (900.0, None, "green", False),
    ]


def test_weather_typing(spark):
    bronze = spark.createDataFrame(
        [(2024, 1, "R", 60.0, 21.0, 35.0, 40.0, 1010.0, "True", 1.2, float("nan"), T0, "b1")],
        "season int, event_round int, session_type string, time double, air_temp double, "
        "track_temp double, humidity double, pressure double, rainfall string, "
        "wind_speed double, wind_direction double, _ingested_at timestamp, _batch_id string",
    )
    row = fastf1.transform_weather(bronze).first()
    assert row.track_temp_c == 35.0 and row.is_raining is True
    assert row.wind_direction_deg is None


def test_car_data_keeps_latest_session_batch(spark):
    schema = (
        "season int, event_round int, session_type string, driver_number string, "
        "session_time double, date timestamp, speed double, rpm double, n_gear double, "
        "throttle double, brake string, drs double, source string, "
        "_ingested_at timestamp, _batch_id string"
    )
    old = (2024, 1, "R", "1", 1.0, T0, 200.0, 11000.0, 6.0, 100.0, "False", 0.0, "car", T0, "a")
    new = old[:6] + (210.0,) + old[7:13] + (T0 + dt.timedelta(days=1), "b")
    rows = fastf1.transform_car_data(spark.createDataFrame([old, new], schema)).collect()
    assert len(rows) == 1 and rows[0].speed_kmh == 210.0 and rows[0].gear == 6


def test_telemetry_fact_distance(spark):
    car = spark.createDataFrame(
        [
            (2024, 1, "R", 1, float(t), T0, 360.0, 11000.0, 7, 100.0, False, 0, "b1")
            for t in range(3)
        ],
        "season int, event_round int, session_type string, driver_number int, "
        "session_time_s double, sample_utc timestamp, speed_kmh double, rpm double, gear int, "
        "throttle_pct double, is_braking boolean, drs int, _source_batch_id string",
    )
    pos = spark.createDataFrame(
        [(2024, 1, "R", 1, 0.5, 1.0, 2.0, 0.0)],
        "season int, event_round int, session_type string, driver_number int, "
        "session_time_s double, x_m double, y_m double, z_m double",
    )
    rows = lap_facts.build_fact_telemetry_sample(car, pos).orderBy("session_time_s").collect()
    assert [r.distance_m for r in rows] == [0.0, 100.0, 200.0]
    assert rows[0].x_m is None and rows[1].x_m == 1.0


def test_driver_id_map_one_to_one(spark):
    results = spark.createDataFrame(
        [(2024, 1, "R", "max_verstappen", "red_bull", 1), (2024, 1, "R", "perez", "red_bull", 11)],
        "season int, round int, session_type string, driver_id string, constructor_id string, "
        "car_number int",
    )
    quali = results.drop("session_type")
    drivers = spark.createDataFrame([("max_verstappen", "VER")], "driver_id string, code string")
    mapping = build_driver_id_map(results, quali, drivers)
    check_one_to_one(mapping)
    assert {(r.driver_number, r.code) for r in mapping.collect()} == {(1, "VER"), (11, None)}

    renamed = F.concat(F.col("driver_id"), F.lit("_x"))
    clash = mapping.unionByName(mapping.limit(1).withColumn("driver_id", renamed))
    with pytest.raises(DriverMapError):
        check_one_to_one(clash)


def test_tied_positions_do_not_duplicate_laps(spark):
    laps = spark.createDataFrame(
        [_lap(1, 1, 0.0, 95.0, 1), _lap(11, 1, 0.0, 95.5, 1), _lap(44, 1, 0.0, 96.0, 2)],
        LAPS_SCHEMA,
    )
    fact = lap_facts.build_fact_lap(laps, _weather(spark), _map(spark))
    assert fact.count() == 3
    check_unique(fact, "lap_key")
    gap = fact.filter("driver_number = 44").first().gap_to_car_ahead_s
    assert gap == pytest.approx(1.0)
