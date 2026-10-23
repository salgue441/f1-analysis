import datetime as dt

from pitwall.silver.laps import transform_laps

BRONZE_SCHEMA = (
    "season int, event_round int, session_type string, driver string, driver_number string, "
    "team string, lap_number double, stint double, lap_time double, sector1_time double, "
    "sector2_time double, sector3_time double, compound string, tyre_life double, "
    "fresh_tyre string, pit_in_time double, pit_out_time double, track_status string, "
    "is_accurate string, deleted string, position double, "
    "_ingested_at timestamp, _batch_id string"
)


def _row(lap_time, ingested_at, pit_in=float("nan"), track_status="1"):
    return (
        2024,
        1,
        "R",
        "VER",
        "1",
        "Red Bull Racing",
        10.0,
        1.0,
        lap_time,
        30.0,
        40.0,
        lap_time - 70.0,
        "soft",
        9.0,
        "True",
        pit_in,
        float("nan"),
        track_status,
        "True",
        "False",
        1.0,
        ingested_at,
        "b1",
    )


def test_dedup_keeps_latest_ingestion_and_types_columns(spark):
    old, new = dt.datetime(2024, 3, 1), dt.datetime(2024, 3, 2)
    bronze = spark.createDataFrame([_row(95.0, old), _row(94.5, new)], BRONZE_SCHEMA)

    rows = transform_laps(bronze).collect()

    assert len(rows) == 1
    lap = rows[0]
    assert lap.lap_time_s == 94.5
    assert lap.compound == "SOFT"
    assert lap.driver_number == 1 and lap.lap_number == 10
    assert lap.fresh_tyre is True
    assert lap.is_pit_in_lap is False  # NaN pit time means no pit
    assert lap.is_green_flag is True


def test_pit_and_flag_detection(spark):
    bronze = spark.createDataFrame(
        [_row(110.0, dt.datetime(2024, 3, 1), pit_in=3600.0, track_status="14")], BRONZE_SCHEMA
    )
    lap = transform_laps(bronze).first()
    assert lap.is_pit_in_lap is True
    assert lap.is_green_flag is False
