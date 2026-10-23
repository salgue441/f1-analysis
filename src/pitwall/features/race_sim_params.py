"""feat_race_sim_params (RQ3): simulator inputs per race, using only pre-race information.

- field and grid: race grid from Jolpica (pit-lane starts go to the back);
- pace: qualifying gap to pole, the standard pre-race pace signal;
- pit loss: the circuit's median (in-lap + out-lap - 2 x clean lap) from *earlier seasons*,
  shrunk to the all-circuit median when there is little history;
- SC/VSC hazard per lap: the circuit's neutralization starts per lap from earlier seasons,
  shrunk toward the global rate (gamma-Poisson style pseudo-count);
- degradation and fuel effect: RQ1 posterior draws for the circuit (circuits RQ1 never saw
  use the per-draw mean across circuits);
- strategies: what each driver actually ran (fact_stint), for the backtest and as the
  baseline in the decision analysis.

Pandas in, `RaceSetup` out: these are small per-race tables (ADR-0003 handoff boundary).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitwall.simulation.distributed import RaceSetup
from pitwall.simulation.field import Car, FieldParams
from pitwall.simulation.strategies import strategy_from_stints

DEFAULT_SC_HAZARD = 0.02
SC_PSEUDO_LAPS = 300.0
PIT_LOSS_PSEUDO_STOPS = 10.0


@dataclass(frozen=True)
class RaceInputs:
    setup: RaceSetup
    driver_ids: list[str]
    actual_finish: np.ndarray  # 1-based classified order (retirements at the back)
    grid: np.ndarray
    circuit_id: str


def pit_loss_by_circuit(pit_losses: pd.DataFrame, before_season: int) -> tuple[dict, float, float]:
    """Shrunk median pit loss per circuit from seasons before `before_season`."""
    prior = pit_losses[
        (pit_losses["season"] < before_season) & pit_losses["pit_loss_s"].between(5, 60)
    ]
    if prior.empty:
        return {}, 22.0, 1.5
    overall = float(prior["pit_loss_s"].median())
    sd = float(np.clip(prior.groupby("circuit_id")["pit_loss_s"].std().median(), 0.5, 3.0))
    g = prior.groupby("circuit_id")["pit_loss_s"].agg(["median", "size"])
    w = g["size"] / (g["size"] + PIT_LOSS_PSEUDO_STOPS)
    return (w * g["median"] + (1 - w) * overall).to_dict(), overall, sd


def sc_hazard_by_circuit(sc_events: pd.DataFrame, before_season: int) -> tuple[dict, float]:
    """Neutralization starts per lap, shrunk toward the global rate of earlier seasons."""
    prior = sc_events[sc_events["season"] < before_season]
    if prior.empty or prior["laps"].sum() == 0:
        return {}, DEFAULT_SC_HAZARD
    global_rate = float(prior["n_sc_starts"].sum() / prior["laps"].sum())
    g = prior.groupby("circuit_id")[["n_sc_starts", "laps"]].sum()
    rate = (g["n_sc_starts"] + SC_PSEUDO_LAPS * global_rate) / (g["laps"] + SC_PSEUDO_LAPS)
    return rate.to_dict(), global_rate


def degradation_draws(posterior: pd.DataFrame, circuit_id: str) -> tuple[dict, np.ndarray]:
    part = posterior[posterior["circuit_id"] == circuit_id]
    if part.empty:  # circuit unseen by RQ1: average over circuits per draw
        part = posterior.groupby(["compound", "draw"], as_index=False)[
            ["beta_s_per_lap", "fuel_effect_s_per_lap"]
        ].mean()
    deg = {
        k: g.sort_values("draw")["beta_s_per_lap"].to_numpy() for k, g in part.groupby("compound")
    }
    fuel = part.groupby("draw")["fuel_effect_s_per_lap"].first().sort_index().to_numpy()
    return deg, fuel


def build_race_inputs(
    season: int,
    sessions: pd.DataFrame,
    results: pd.DataFrame,
    stints: pd.DataFrame,
    pit_losses: pd.DataFrame,
    sc_events: pd.DataFrame,
    posterior: pd.DataFrame,
) -> list[RaceInputs]:
    loss_by_circuit, loss_overall, loss_sd = pit_loss_by_circuit(pit_losses, season)
    hazard_by_circuit, hazard_global = sc_hazard_by_circuit(sc_events, season)
    races = sessions[(sessions["season"] == season) & (sessions["session_type"] == "R")]
    out = []
    for race in races.sort_values("round").itertuples(index=False):
        rnd, circuit, laps = int(race.round), race.circuit_id, int(race.scheduled_laps or 0)
        in_race = (results["season"] == season) & (results["round"] == rnd)
        r = results[in_race & (results["session_type"] == "R")].sort_values("finish_position")
        if r.empty or laps < 10:
            continue
        q = results[in_race & (results["session_type"] == "Q")].set_index("driver_id")
        pole = q["best_lap_time_s"].min()
        gaps = (q["best_lap_time_s"] - pole).reindex(r["driver_id"])
        # No qualifying time: half a second behind the slowest qualifier.
        gaps = gaps.fillna((gaps.max() if gaps.notna().any() else 0.0) + 0.5).clip(upper=5.0)

        race_stints = stints[(stints["season"] == season) & (stints["event_round"] == rnd)]
        cars = []
        for row, gap in zip(r.itertuples(index=False), gaps.to_numpy(), strict=True):
            own = race_stints[race_stints["driver_id"] == row.driver_id]
            cars.append(
                Car(
                    driver_id=row.driver_id,
                    grid=int(row.grid) if pd.notna(row.grid) else 0,
                    pace_offset_s=float(gap),
                    strategy=strategy_from_stints(own, laps),
                )
            )
        deg, fuel = degradation_draws(posterior, circuit)
        for compound in ("SOFT", "MEDIUM", "HARD"):
            deg.setdefault(compound, np.full(len(fuel), 0.05))
        params = FieldParams(
            race_laps=laps,
            base_lap_s=float(pole * 1.07) if pd.notna(pole) else 95.0,
            deg_draws=deg,
            fuel_draws=fuel,
            pit_loss_mean_s=float(loss_by_circuit.get(circuit, loss_overall)),
            pit_loss_sd_s=loss_sd,
            sc_hazard_per_lap=float(hazard_by_circuit.get(circuit, hazard_global)),
        )
        race_id = f"{season}_{rnd:02d}"
        out.append(
            RaceInputs(
                setup=RaceSetup(race_id, tuple(cars), params),
                driver_ids=list(r["driver_id"]),
                actual_finish=np.arange(1, len(r) + 1),
                grid=np.array([c.grid for c in cars]),
                circuit_id=circuit,
            )
        )
    return out
