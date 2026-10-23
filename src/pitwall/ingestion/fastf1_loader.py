"""Load session data via FastF1 into bronze-ready pandas DataFrames.

Bronze keeps FastF1's columns (snake_cased) with timedeltas converted to seconds.
Object columns are stored as strings; silver is responsible for typing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    for col in out.columns:
        if pd.api.types.is_timedelta64_dtype(out[col]):
            out[col] = out[col].dt.total_seconds()
        elif out[col].dtype == object:
            out[col] = out[col].astype("string")
    out.columns = [to_snake(c) for c in out.columns]
    return out


def event_rounds(season: int) -> list[int]:
    import fastf1

    schedule = fastf1.get_event_schedule(season, include_testing=False)
    return [int(r) for r in schedule["RoundNumber"].tolist()]


def load_session_laps(season: int, round_: int, session_type: str, cache_dir: str) -> pd.DataFrame:
    import fastf1

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(season, round_, session_type)
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = normalize_frame(session.laps)
    laps["season"] = season
    laps["event_round"] = round_
    laps["session_type"] = session_type
    return laps
