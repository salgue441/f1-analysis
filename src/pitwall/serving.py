"""Read-side helpers for the Streamlit app: Delta tables via delta-rs, no Spark/JVM.

Everything degrades gracefully: a missing table returns None, and `demo_setup` gives the
app a self-contained race to simulate before any data has been loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitwall.config import get_settings
from pitwall.paths import table_path
from pitwall.simulation.distributed import RaceSetup
from pitwall.simulation.field import Car, FieldParams
from pitwall.simulation.race_simulator import Stint, Strategy

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def read_table(layer: str, table: str, filters: list | None = None) -> pd.DataFrame | None:
    try:
        from deltalake import DeltaTable

        dt = DeltaTable(table_path(layer, table, get_settings().lakehouse_root))  # type: ignore[arg-type]
        return dt.to_pandas(filters=filters)
    except Exception:  # noqa: BLE001 - table not built yet (or no deltalake): app shows a hint
        return None


def race_options() -> pd.DataFrame | None:
    """Races that have simulator inputs, labelled with the event name."""
    races = read_table("features", "feat_race_sim_params")
    sessions = read_table("gold", "dim_session")
    if races is None:
        return None
    if sessions is not None:
        names = sessions[sessions["session_type"] == "R"].assign(
            race_id=lambda d: (
                d["season"].astype(str) + "_" + d["round"].astype(int).map("{:02d}".format)
            )
        )[["race_id", "event_name"]]
        races = races.merge(names, on="race_id", how="left")
    else:
        races["event_name"] = races["race_id"]
    return races.sort_values("race_id")


def load_setup(race_id: str) -> RaceSetup | None:
    from pitwall.features.race_sim_params import from_tables

    races = read_table("features", "feat_race_sim_params")
    cars = read_table("features", "feat_race_sim_cars")
    posterior = read_table("features", "rq1_degradation_posterior")
    if races is None or cars is None or posterior is None:
        return None
    race = races[races["race_id"] == race_id].iloc[0]
    return from_tables(race, cars[cars["race_id"] == race_id], posterior)


def demo_setup(seed: int = 0) -> RaceSetup:
    """A plausible 20-car, 57-lap race with made-up parameters (clearly labelled in the app)."""
    rng = np.random.default_rng(seed)
    laps = 57
    deg = {
        "SOFT": rng.normal(0.085, 0.01, 1000),
        "MEDIUM": rng.normal(0.05, 0.008, 1000),
        "HARD": rng.normal(0.03, 0.006, 1000),
    }
    params = FieldParams(
        race_laps=laps,
        base_lap_s=96.0,
        deg_draws=deg,
        fuel_draws=rng.normal(0.035, 0.004, 1000),
        pit_loss_mean_s=22.0,
        pit_loss_sd_s=1.2,
        sc_hazard_per_lap=0.015,
    )
    one_stop = Strategy("MH-20", (Stint("MEDIUM", 20), Stint("HARD", laps - 20)))
    two_stop = Strategy("SHM-15-38", (Stint("SOFT", 15), Stint("HARD", 23), Stint("MEDIUM", 19)))
    cars = tuple(
        Car(
            f"driver_{i + 1:02d}",
            i + 1,
            0.12 * i + rng.normal(0, 0.05),
            two_stop if i % 3 == 0 else one_stop,
        )
        for i in range(20)
    )
    return RaceSetup("demo", cars, params)


def laps_for_race(race_id: str) -> pd.DataFrame | None:
    season, rnd = (int(x) for x in race_id.split("_"))
    return read_table(
        "gold",
        "fact_lap",
        filters=[("season", "=", season), ("event_round", "=", rnd), ("session_type", "=", "R")],
    )


def live_state() -> pd.DataFrame | None:
    """stream.race_state for the most recently updated session (`pitwall stream ...`)."""
    state = read_table("stream", "race_state")
    if state is None or state.empty:
        return None
    latest = state.loc[state["_updated_at"].idxmax(), "session_key"]
    return state[state["session_key"] == latest].sort_values("position")


def findings() -> dict[str, dict]:
    out = {}
    for rq in ("rq1", "rq2", "rq3", "rq4"):
        path = REPORTS / rq / "results.json"
        if path.exists():
            out[rq] = json.loads(path.read_text())
    return out
