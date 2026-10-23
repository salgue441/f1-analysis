"""Load session data via FastF1 into bronze-ready pandas DataFrames.

Bronze keeps FastF1's columns (snake_cased) with timedeltas converted to seconds.
Object columns are stored as strings; silver is responsible for typing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Splits camelCase and acronyms: "LapStartTime" -> lap_start_time, "RPM" -> rpm,
# "nGear" -> n_gear, "SpeedI1" -> speed_i1.
_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Session-level frames: bronze table -> attribute of a loaded fastf1 Session.
SESSION_FRAMES = {
    "fastf1_laps": "laps",
    "fastf1_weather": "weather_data",
    "fastf1_track_status": "track_status",
}
# Per-driver telemetry: bronze table -> attribute (a dict of driver number -> frame).
TELEMETRY_FRAMES = {
    "fastf1_car_data": "car_data",
    "fastf1_pos_data": "pos_data",
}


def to_snake(name: str) -> str:
    return _WORD_BOUNDARY.sub("_", name).lower()


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    for col in out.columns:
        if pd.api.types.is_timedelta64_dtype(out[col]):
            out[col] = out[col].dt.total_seconds()
        elif out[col].dtype == object:
            out[col] = out[col].astype("string")
    out.columns = [to_snake(c) for c in out.columns]
    return out


def _tag(df: pd.DataFrame, season: int, round_: int, session_type: str) -> pd.DataFrame:
    df["season"] = season
    df["event_round"] = round_
    df["session_type"] = session_type
    return df


def event_rounds(season: int) -> list[int]:
    import fastf1

    schedule = fastf1.get_event_schedule(season, include_testing=False)
    return [int(r) for r in schedule["RoundNumber"].tolist()]


def _load(season: int, round_: int, session_type: str, cache_dir: str, telemetry: bool):
    import fastf1

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(season, round_, session_type)
    session.load(laps=True, telemetry=telemetry, weather=True, messages=True)
    return session


def load_session(
    season: int, round_: int, session_type: str, cache_dir: str, telemetry: bool = False
) -> dict[str, pd.DataFrame]:
    """All bronze frames for one session, keyed by bronze table name."""
    session = _load(season, round_, session_type, cache_dir, telemetry)
    frames = {
        table: _tag(normalize_frame(getattr(session, attr)), season, round_, session_type)
        for table, attr in SESSION_FRAMES.items()
    }
    if telemetry:
        for table, attr in TELEMETRY_FRAMES.items():
            per_driver = [
                normalize_frame(frame).assign(driver_number=str(number))
                for number, frame in getattr(session, attr).items()
            ]
            if per_driver:
                combined = pd.concat(per_driver, ignore_index=True)
                frames[table] = _tag(combined, season, round_, session_type)
    return frames


def load_session_laps(season: int, round_: int, session_type: str, cache_dir: str) -> pd.DataFrame:
    return load_session(season, round_, session_type, cache_dir)["fastf1_laps"]
