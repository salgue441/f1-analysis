import datetime as dt
import json

import pandas as pd
import pytest

from pitwall import config
from pitwall.streaming.landing import LandingWriter
from pitwall.streaming.live import LiveBuffer
from pitwall.streaming.replay import release_times, replay
from pitwall.streaming.state import StrategyContext, build_race_state

T0 = dt.datetime(2025, 12, 7, 13, 0, tzinfo=dt.UTC)
KEY = 9999


def _iso(seconds: float) -> str:
    return (T0 + dt.timedelta(seconds=seconds)).isoformat()


def _lap(driver, lap, start_s, duration=90.0):
    return {
        "session_key": KEY,
        "driver_number": driver,
        "lap_number": lap,
        "date_start": _iso(start_s),
        "lap_duration": duration,
    }


def _race_payloads(status_message="TRACK CLEAR"):
    """Two cars, 12 laps each; car 1 leads car 4 by 1.5 s; car 4 pits once (lap 2)."""
    laps = []
    for lap in range(1, 13):
        laps.append(_lap(1, lap, 90.0 * (lap - 1)))
        laps.append(_lap(4, lap, 90.0 * (lap - 1) + 1.5))
    return {
        "drivers": [
            {"session_key": KEY, "driver_number": 1, "name_acronym": "VER", "team_name": "RB"},
            {"session_key": KEY, "driver_number": 4, "name_acronym": "NOR", "team_name": "McL"},
        ],
        "laps": laps,
        "stints": [
            {
                "session_key": KEY,
                "driver_number": 1,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 12,
                "compound": "MEDIUM",
                "tyre_age_at_start": 0,
            },
            {
                "session_key": KEY,
                "driver_number": 4,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 2,
                "compound": "SOFT",
                "tyre_age_at_start": 0,
            },
            {
                "session_key": KEY,
                "driver_number": 4,
                "stint_number": 2,
                "lap_start": 3,
                "lap_end": 12,
                "compound": "HARD",
                "tyre_age_at_start": 0,
            },
        ],
        "pit": [{"session_key": KEY, "driver_number": 4, "lap_number": 2, "date": _iso(170)}],
        "position": [
            {"session_key": KEY, "driver_number": 1, "date": _iso(0), "position": 1},
            {"session_key": KEY, "driver_number": 4, "date": _iso(0), "position": 2},
        ],
        "race_control": [
            {
                "session_key": KEY,
                "date": _iso(600),
                "category": "Flag",
                "flag": "CLEAR",
                "scope": "Track",
                "message": status_message,
            },
        ],
    }


def test_records_are_released_when_they_would_have_been_published():
    payloads = _race_payloads()
    payloads["laps"].append({**_lap(1, 13, 1080.0), "lap_duration": None})
    payloads["laps"].append(_lap(1, 14, 1175.0))
    events = release_times(payloads)
    assert [e.endpoint for e in events[:2]] == ["drivers", "drivers"]
    lap1 = next(e for e in events if e.endpoint == "laps")
    assert lap1.at == T0 + dt.timedelta(seconds=90)  # lap 1 ends, not starts
    no_duration = next(e for e in events if e.endpoint == "laps" and e.record["lap_number"] == 13)
    assert no_duration.at == T0 + dt.timedelta(seconds=1175)  # next lap's start
    hard = next(e for e in events if e.endpoint == "stints" and e.record["stint_number"] == 2)
    assert hard.at == T0 + dt.timedelta(seconds=181.5)  # start of lap 3
    assert [e.at for e in events] == sorted(e.at for e in events)


def test_replay_runs_on_an_accelerated_clock(tmp_path):
    events = release_times(_race_payloads())
    now = [0.0]
    writer = LandingWriter(tmp_path)
    files = replay(
        events,
        writer,
        speed=100.0,
        flush_every_s=1.0,
        clock=lambda: now[0],
        sleep=lambda s: now.__setitem__(0, now[0] + s),
    )
    # The race lasts ~1,100 s: about 11 one-second flushes at 100x.
    assert 10 <= files <= 13
    assert writer.records_written == len(events)
    lines = [
        json.loads(x) for f in sorted(tmp_path.glob("*.json")) for x in f.read_text().splitlines()
    ]
    assert len(lines) == len(events) and all("_emitted_at" in x for x in lines)
    assert not list(tmp_path.glob("_*"))  # no half-written files left


def _events_df(spark, payloads):
    rows = [(KEY, endpoint, json.dumps(rec)) for endpoint, recs in payloads.items() for rec in recs]
    return spark.createDataFrame(rows, "session_key long, endpoint string, payload string")


CTX = StrategyContext(
    race_laps=30, deg_s_per_lap={"MEDIUM": 0.06, "HARD": 0.04}, tier="high", undercut_effect_pp=49.5
)


def test_race_state_tyres_gaps_and_undercut_window(spark):
    state = build_race_state(_events_df(spark, _race_payloads()), CTX).toPandas()
    ver, nor = state.set_index("driver").loc["VER"], state.set_index("driver").loc["NOR"]
    assert ver["position"] == 1 and nor["position"] == 2
    assert ver["tyre_age_laps"] == 12 and nor["tyre_age_laps"] == 10
    assert nor["compound"] == "HARD" and nor["pit_stops"] == 1 and ver["pit_stops"] == 0
    assert nor["gap_ahead_s"] == pytest.approx(1.5) and ver["gap_ahead_s"] != ver["gap_ahead_s"]
    assert ver["time_loss_vs_new_s"] == pytest.approx(0.72)
    assert ver["laps_remaining"] == 18 and ver["track_status"] == "green"
    assert bool(nor["undercut_window"]) and nor["undercut_effect_pp"] == pytest.approx(49.5)
    assert not bool(ver["undercut_window"])  # nobody ahead


def test_no_undercut_window_under_safety_car_or_at_low_wear_tracks(spark):
    sc = _race_payloads()
    sc["race_control"].append(
        {
            "session_key": KEY,
            "date": _iso(900),
            "category": "SafetyCar",
            "flag": None,
            "scope": None,
            "message": "SAFETY CAR DEPLOYED",
        }
    )
    state = build_race_state(_events_df(spark, sc), CTX).toPandas()
    assert set(state["track_status"]) == {"safety_car"} and not state["undercut_window"].any()
    low = StrategyContext(30, {}, tier="low")
    state = build_race_state(_events_df(spark, _race_payloads()), low).toPandas()
    assert not state["undercut_window"].any()


@pytest.fixture
def lakehouse(tmp_path, monkeypatch):
    monkeypatch.setenv("PITWALL_LAKEHOUSE_ROOT", str(tmp_path / "lakehouse"))
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


def test_stream_end_to_end_is_exactly_once(spark, lakehouse):
    from pitwall.paths import table_path
    from pitwall.streaming.pipeline import StreamSession, landing_dir, start_stream

    session = StreamSession(KEY, 2025, 24)
    events = release_times(_race_payloads())
    writer = LandingWriter(landing_dir(KEY))
    half = len(events) // 2
    writer.write([(e.endpoint, e.record) for e in events[:half]])

    def run():
        start_stream(spark, session, CTX, available_now=True).awaitTermination()

    def read(table):
        return spark.read.format("delta").load(table_path("stream", table)).toPandas()

    run()
    assert len(read("events")) == half
    first = read("race_state")
    assert first["lap_number"].max() < 12

    writer.write([(e.endpoint, e.record) for e in events[half:]])
    run()
    run()  # nothing new: no batch, no duplicates
    assert len(read("events")) == len(events)
    state = read("race_state")
    assert len(state) == 2 and state["lap_number"].max() == 12
    batches = read("_batches")
    assert len(batches) == 2 and (batches["max_latency_s"] >= 0).all()


def test_reset_then_replay_reprocesses_everything_in_one_running_query(spark, lakehouse):
    """Regression: batch ids restart with a new checkpoint, so Delta's idempotency id must
    too, or the new run's batches are skipped as already committed. Also exercises the
    in-memory session log across batches of one long-running query."""
    from pitwall.paths import table_path
    from pitwall.streaming.pipeline import StreamSession, landing_dir, reset_session, start_stream

    session = StreamSession(KEY, 2025, 24)
    events = release_times(_race_payloads())

    def stream_in_chunks():
        writer = LandingWriter(landing_dir(KEY))
        query = start_stream(spark, session, CTX, trigger_s=0.5)
        for chunk in (events[:10], events[10:30], events[30:]):
            writer.write([(e.endpoint, e.record) for e in chunk])
            query.processAllAvailable()
        query.stop()

    def read(table):
        return spark.read.format("delta").load(table_path("stream", table)).toPandas()

    stream_in_chunks()
    first = read("race_state").sort_values("driver_number").reset_index(drop=True)
    reset_session(KEY)
    stream_in_chunks()
    assert len(read("events")) == len(events)
    second = read("race_state").sort_values("driver_number").reset_index(drop=True)
    cols = ["driver", "position", "lap_number", "tyre_age_laps", "gap_ahead_s", "undercut_window"]
    pd.testing.assert_frame_equal(first[cols], second[cols])
    assert second["lap_number"].max() == 12


def test_live_buffer_routes_topics_and_filters_sessions(tmp_path):
    buffer = LiveBuffer(LandingWriter(tmp_path), session_key=KEY)
    buffer.on_message("v1/laps", json.dumps(_lap(1, 1, 0)).encode())
    buffer.on_message("v1/laps", json.dumps({**_lap(1, 1, 0), "session_key": 1}).encode())
    buffer.on_message("v1/car_data", b'{"session_key": 9999}')  # not a state endpoint
    buffer.on_message("v1/pit", b"not json")
    assert buffer.flush() == 1
    (file,) = tmp_path.glob("*.json")
    assert json.loads(file.read_text())["_endpoint"] == "laps"
    assert buffer.flush() == 0 and len(list(tmp_path.glob("*.json"))) == 1
