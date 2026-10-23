import datetime as dt
import json

import pytest
from pyspark.sql import functions as F

from pitwall.silver import jolpica
from pitwall.silver.common import duration_s

BRONZE_SCHEMA = "payload string, _ingested_at timestamp, _batch_id string"

VER = {
    "driverId": "max_verstappen",
    "permanentNumber": "33",
    "code": "VER",
    "givenName": "Max",
    "familyName": "Verstappen",
    "dateOfBirth": "1997-09-30",
    "nationality": "Dutch",
}
PER = {**VER, "driverId": "perez", "permanentNumber": "11", "code": "PER", "familyName": "Pérez"}
RBR = {"constructorId": "red_bull", "name": "Red Bull", "nationality": "Austrian"}
CIRCUIT = {
    "circuitId": "bahrain",
    "circuitName": "Bahrain International Circuit",
    "Location": {"lat": "26.0325", "long": "50.5106", "locality": "Sakhir", "country": "Bahrain"},
}


def _race(**tables):
    return {
        "season": "2024",
        "round": "1",
        "raceName": "Bahrain Grand Prix",
        "Circuit": CIRCUIT,
        "date": "2024-03-02",
        "time": "15:00:00Z",
        "Qualifying": {"date": "2024-03-01", "time": "16:00:00Z"},
        **tables,
    }


def _page(races=None, **tables):
    mr = {"total": "2", "RaceTable": {"Races": races or []}, **tables}
    return json.dumps({"MRData": mr})


def _result(driver, position, text, points, grid="1", millis="5504742"):
    return {
        "number": driver["permanentNumber"],
        "position": position,
        "positionText": text,
        "points": points,
        "Driver": driver,
        "Constructor": RBR,
        "grid": grid,
        "laps": "57",
        "status": "Finished",
        "Time": {"millis": millis, "time": "1:31:44.742"},
        "FastestLap": {"rank": "1", "lap": "39", "Time": {"time": "1:32.608"}},
    }


def _bronze(spark, *pages, ingested=dt.datetime(2024, 3, 4)):
    return spark.createDataFrame([(p, ingested, "b1") for p in pages], BRONZE_SCHEMA)


def test_duration_parsing(spark):
    df = spark.createDataFrame(
        [("1:32.608",), ("22.123",), ("1:02:03.5",), ("",), (None,), ("abc",)], "t string"
    )
    got = [r.s for r in df.select(duration_s(F.col("t")).alias("s")).collect()]
    assert got[:3] == [pytest.approx(92.608), pytest.approx(22.123), pytest.approx(3723.5)]
    assert got[3:] == [None, None, None]


def test_results_across_pages_are_exploded_and_typed(spark):
    # Pagination splits one race's results over two pages.
    page1 = _page([_race(Results=[_result(VER, "1", "1", "26")])])
    page2 = _page([_race(Results=[_result(PER, "20", "R", "0", grid="5", millis=None)])])

    rows = {
        r.driver_id: r for r in jolpica.transform_results(_bronze(spark, page1, page2)).collect()
    }

    assert set(rows) == {"max_verstappen", "perez"}
    ver, per = rows["max_verstappen"], rows["perez"]
    assert (ver.season, ver.round, ver.session_type, ver.car_number) == (2024, 1, "R", 33)
    assert ver.points == 26.0 and ver.is_classified is True
    assert ver.race_time_s == pytest.approx(5504.742)
    assert ver.fastest_lap_time_s == pytest.approx(92.608)
    assert per.is_classified is False and per.finish_position == 20 and per.race_time_s is None


def test_results_dedup_keeps_latest_ingestion(spark):
    old = _bronze(spark, _page([_race(Results=[_result(VER, "1", "1", "25")])]))
    new = _bronze(
        spark,
        _page([_race(Results=[_result(VER, "1", "1", "26")])]),
        ingested=dt.datetime(2024, 3, 5),
    )
    rows = jolpica.transform_results(old.unionByName(new)).collect()
    assert len(rows) == 1 and rows[0].points == 26.0


def test_sprint_results_get_session_type_s(spark):
    page = _page([_race(SprintResults=[_result(VER, "1", "1", "8")])])
    row = jolpica.transform_sprint_results(_bronze(spark, page)).first()
    assert row.session_type == "S" and row.points == 8.0


def test_races_have_utc_session_starts(spark):
    row = jolpica.transform_races(_bronze(spark, _page([_race()]))).first()
    assert row.circuit_id == "bahrain"
    assert row.race_date == dt.date(2024, 3, 2)
    assert row.race_start_utc == dt.datetime(2024, 3, 2, 15, 0)
    assert row.q_start_utc == dt.datetime(2024, 3, 1, 16, 0)
    assert row.s_start_utc is None


def test_qualifying_best_time(spark):
    quali = {
        "number": "33",
        "position": "1",
        "Driver": VER,
        "Constructor": RBR,
        "Q1": "1:30.031",
        "Q2": "1:29.374",
        "Q3": "1:29.179",
    }
    knocked_out = {**quali, "Driver": PER, "position": "18", "Q2": "", "Q3": ""}
    page = _page([_race(QualifyingResults=[quali, knocked_out])])
    rows = {r.driver_id: r for r in jolpica.transform_qualifying(_bronze(spark, page)).collect()}
    assert rows["max_verstappen"].best_time_s == pytest.approx(89.179)
    assert rows["perez"].q3_s is None and rows["perez"].best_time_s == pytest.approx(90.031)


def test_pit_stops(spark):
    stops = [
        {"driverId": "perez", "lap": "17", "stop": "1", "time": "15:30:00", "duration": "23.4"},
        {"driverId": "perez", "lap": "30", "stop": "2", "time": "15:50:00", "duration": "1:01.2"},
    ]
    rows = jolpica.transform_pit_stops(_bronze(spark, _page([_race(PitStops=stops)]))).collect()
    assert sorted((r.stop_number, r.lap, r.duration_s) for r in rows) == [
        (1, 17, pytest.approx(23.4)),
        (2, 30, pytest.approx(61.2)),
    ]


def test_entity_tables(spark):
    page = _page(
        DriverTable={"Drivers": [VER, PER]},
        ConstructorTable={"Constructors": [RBR]},
        CircuitTable={"Circuits": [CIRCUIT]},
    )
    bronze = _bronze(spark, page)

    drivers = {r.driver_id: r for r in jolpica.transform_drivers(bronze).collect()}
    assert drivers["max_verstappen"].permanent_number == 33
    assert drivers["max_verstappen"].date_of_birth == dt.date(1997, 9, 30)

    constructor = jolpica.transform_constructors(bronze).first()
    assert constructor.constructor_name == "Red Bull"

    circuit = jolpica.transform_circuits(bronze).first()
    assert circuit.lat == pytest.approx(26.0325) and circuit.country == "Bahrain"
