import datetime as dt
import json

import pytest

from pitwall.pipelines import match_rounds
from pitwall.silver import openf1
from pitwall.silver.laps import transform_laps as fastf1_transform_laps

START = dt.datetime(2024, 3, 2, 15, 0)
T0 = dt.datetime(2024, 3, 4)
BRONZE_SCHEMA = (
    "season int, event_round int, session_type string, session_key int, "
    "session_start_utc timestamp, driver_number int, payload string, url string, "
    "_ingested_at timestamp, _batch_id string"
)


def _at(seconds: float) -> str:
    return (START + dt.timedelta(seconds=seconds)).isoformat() + "+00:00"


def _bronze(spark, records, ingested=T0):
    row = (2024, 1, "R", 9472, START, None, json.dumps(records), "u", ingested, "b1")
    return spark.createDataFrame([row], BRONZE_SCHEMA)


def _lap(n, start, duration, pit_out=False, s1=30.0):
    return {
        "driver_number": 1,
        "lap_number": n,
        "date_start": _at(start),
        "lap_duration": duration,
        "duration_sector_1": s1,
        "duration_sector_2": 30.0,
        "duration_sector_3": duration - 30.0 - (s1 or 0),
        "st_speed": 300,
        "is_pit_out_lap": pit_out,
    }


@pytest.fixture
def lap_inputs(spark):
    laps = [
        _lap(1, 0, 100.0, s1=None),
        _lap(2, 100, 95.0),
        _lap(3, 195, 96.0),
        _lap(4, 291, 115.0, pit_out=True),
        _lap(5, 406, 94.0),
    ]
    stints = [
        {
            "driver_number": 1,
            "stint_number": 1,
            "lap_start": 1,
            "lap_end": 3,
            "compound": "SOFT",
            "tyre_age_at_start": 0,
        },
        {
            "driver_number": 1,
            "stint_number": 2,
            "lap_start": 4,
            "lap_end": 4,
            "compound": "HARD",
            "tyre_age_at_start": 3,
        },
    ]
    pit = [{"driver_number": 1, "lap_number": 3, "pit_duration": 22.0, "date": _at(280)}]
    position = [
        {"driver_number": 1, "date": _at(-60), "position": 2},
        {"driver_number": 1, "date": _at(150), "position": 1},
    ]
    race_control = [
        {
            "date": _at(-120),
            "lap_number": 1,
            "category": "Flag",
            "flag": "GREEN",
            "scope": "Track",
            "driver_number": None,
            "message": "GREEN LIGHT - PIT EXIT OPEN",
        },
        {
            "date": _at(200),
            "lap_number": 3,
            "category": "SafetyCar",
            "flag": None,
            "scope": None,
            "driver_number": None,
            "message": "VIRTUAL SAFETY CAR DEPLOYED",
        },
        {
            "date": _at(260),
            "lap_number": 3,
            "category": "SafetyCar",
            "flag": None,
            "scope": None,
            "driver_number": None,
            "message": "VIRTUAL SAFETY CAR ENDING",
        },
        {
            "date": _at(280),
            "lap_number": 3,
            "category": "Flag",
            "flag": "CLEAR",
            "scope": "Track",
            "driver_number": None,
            "message": "TRACK CLEAR",
        },
        {
            "date": _at(300),
            "lap_number": 4,
            "category": "Flag",
            "flag": "CLEAR",
            "scope": "Track",
            "driver_number": None,
            "message": "TRACK CLEAR",
        },
        {
            "date": _at(410),
            "lap_number": 5,
            "category": "Other",
            "flag": None,
            "scope": None,
            "driver_number": 1,
            "message": "CAR 1 (VER) TIME 1:35.000 DELETED - TRACK LIMITS AT TURN 4 LAP 2 15:06:50",
        },
    ]
    drivers = [{"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"}]
    return [_bronze(spark, x) for x in (laps, stints, pit, position, race_control, drivers)]


def test_race_control_periods_collapse_repeats(spark, lap_inputs):
    rows = openf1.transform_race_control(lap_inputs[4]).orderBy("start_s").collect()
    assert [(r.start_s, r.status) for r in rows] == [
        (-120.0, "1"),
        (200.0, "6"),
        (260.0, "7"),
        (280.0, "1"),  # the repeated CLEAR at 300 s is not a change
    ]
    assert rows[1].end_s == 260.0 and rows[1].is_neutralized and rows[-1].end_s is None


def test_laps_match_fastf1_schema(spark, lap_inputs):
    openf1_cols = openf1.transform_laps(*lap_inputs).columns
    ff1_schema = (
        "season int, event_round int, session_type string, driver string, "
        "driver_number string, team string, lap_number double, stint double, "
        "lap_time double, sector1_time double, sector2_time double, sector3_time double, "
        "compound string, tyre_life double, fresh_tyre string, pit_in_time double, "
        "pit_out_time double, track_status string, is_accurate string, deleted string, "
        "position double, lap_start_time double, time double, speed_st double, "
        "_ingested_at timestamp, _batch_id string"
    )
    ff1_cols = fastf1_transform_laps(spark.createDataFrame([], ff1_schema)).columns
    assert openf1_cols == ff1_cols


def test_laps_derived_fields(spark, lap_inputs):
    laps = {r.lap_number: r for r in openf1.transform_laps(*lap_inputs).collect()}
    assert set(laps) == {1, 2, 3, 4, 5}

    assert (laps[2].compound, laps[2].tyre_life_laps, laps[2].fresh_tyre) == ("SOFT", 2, True)
    assert (laps[4].compound, laps[4].tyre_life_laps, laps[4].fresh_tyre) == ("HARD", 4, False)
    assert laps[5].compound is None  # outside every stint: kept, untyred

    assert laps[3].is_pit_in_lap and laps[4].is_pit_out_lap
    assert laps[2].lap_start_time_s == 100.0 and laps[2].lap_end_time_s == 195.0
    assert laps[1].position_end_of_lap == 2 and laps[2].position_end_of_lap == 1

    assert laps[2].track_status == "1" and laps[2].is_green_flag
    assert laps[3].track_status == "167" and not laps[3].is_green_flag

    assert laps[2].is_deleted and not laps[3].is_deleted
    assert laps[1].is_accurate is False  # missing sector 1
    assert laps[2].is_accurate and not laps[3].is_accurate  # lap 3 is a pit in-lap
    assert laps[2].driver_code == "VER" and laps[2].team_name == "Red Bull Racing"


def test_weather_and_telemetry(spark):
    weather = _bronze(
        spark,
        [
            {
                "date": _at(60),
                "air_temperature": 20.0,
                "track_temperature": 30.0,
                "humidity": 40.0,
                "pressure": 1010.0,
                "rainfall": 1,
                "wind_speed": 1.0,
                "wind_direction": 180,
            }
        ],
    )
    w = openf1.transform_weather(weather).first()
    assert w.session_time_s == 60.0 and w.track_temp_c == 30.0 and w.is_raining is True

    car = _bronze(
        spark,
        [
            {
                "driver_number": 1,
                "date": _at(10.5),
                "speed": 300,
                "rpm": 11000,
                "n_gear": 7,
                "throttle": 99,
                "brake": 100,
                "drs": 12,
            }
        ],
    )
    c = openf1.transform_car_data(car).first()
    assert c.session_time_s == 10.5 and c.is_braking is True and c.gear == 7

    loc = _bronze(spark, [{"driver_number": 1, "date": _at(10.5), "x": 1234, "y": -50, "z": 7}])
    p = openf1.transform_pos_data(loc).first()
    assert (p.x_m, p.y_m) == (123.4, -5.0)


def test_only_latest_batch_per_session_is_used(spark):
    old = _bronze(spark, [{"driver_number": 1, "name_acronym": "OLD", "team_name": "x"}])
    new = _bronze(
        spark,
        [{"driver_number": 1, "name_acronym": "VER", "team_name": "x"}],
        ingested=T0 + dt.timedelta(days=1),
    )
    rows = openf1.explode_payload(old.unionByName(new), "drivers").collect()
    assert [r.name_acronym for r in rows] == ["VER"]


def test_match_rounds_by_nearest_start():
    races = [
        {"round": 1, "race_start_utc": dt.datetime(2024, 3, 2, 15)},
        {"round": 2, "race_start_utc": dt.datetime(2024, 3, 9, 17)},
    ]
    sessions = [
        {"session_key": 10, "date_start": "2024-03-09T17:00:00+00:00"},
        {"session_key": 11, "date_start": "2024-03-02T15:00:00+00:00"},
        {"session_key": 12, "date_start": "2024-06-01T15:00:00+00:00"},  # no Jolpica race
    ]
    assert match_rounds(sessions, races, "R") == {10: 2, 11: 1}
