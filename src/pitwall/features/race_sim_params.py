"""feat_race_sim_params (RQ3): simulator inputs per race, using only pre-race information.

- field and grid: race grid from Jolpica (pit-lane starts go to the back);
- pace: qualifying gap to pole, the standard pre-race pace signal, scaled by `pace_scale`
  plus a per-race deviation (`pace_noise_sd_s`); both are tuned on the previous season;
- pit loss: the circuit's median (in-lap + out-lap - 2 x clean lap) from *earlier seasons*,
  shrunk to the all-circuit median when there is little history;
- SC/VSC hazard per lap: the circuit's neutralization starts per lap from earlier seasons,
  shrunk toward the global rate (gamma-Poisson style pseudo-count);
- retirement hazard per lap: non-classified finishes per lap raced in earlier seasons
  (results back to 2018);
- degradation, compound pace and fuel effect: joint RQ1 posterior draws for the circuit
  (circuits RQ1 never saw use the per-draw mean across circuits);
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
from pitwall.simulation.race_simulator import Stint, Strategy
from pitwall.simulation.strategies import strategy_from_stints

DEFAULT_SC_HAZARD = 0.02
DEFAULT_DNF_HAZARD = 0.0015
SC_PSEUDO_LAPS = 300.0
PIT_LOSS_PSEUDO_STOPS = 10.0
DRY = ("SOFT", "MEDIUM", "HARD")


@dataclass(frozen=True)
class RaceInputs:
    setup: RaceSetup
    driver_ids: list[str]
    actual_finish: np.ndarray  # 1-based classified order (retirements at the back)
    grid: np.ndarray
    circuit_id: str


@dataclass(frozen=True)
class PaceSettings:
    """Simulator settings not estimated by RQ1, tuned on the season before the test season."""

    pace_scale: float = 1.0
    pace_noise_sd_s: float = 0.0


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


def dnf_hazard(results: pd.DataFrame, before_season: int) -> float:
    """Retirements per lap raced (race sessions of earlier seasons)."""
    prior = results[(results["season"] < before_season) & (results["session_type"] == "R")]
    laps = prior["laps_completed"].sum()
    if prior.empty or not laps:
        return DEFAULT_DNF_HAZARD
    return float((~prior["is_classified"].astype(bool)).sum() / laps)


def posterior_draws(posterior: pd.DataFrame, circuit_id: str) -> tuple[dict, dict, np.ndarray]:
    """(degradation, compound pace offset, fuel effect) draws for one circuit."""
    part = posterior[posterior["circuit_id"] == circuit_id]
    if part.empty:  # circuit unseen by RQ1: average over circuits per draw
        part = posterior.groupby(["compound", "draw"], as_index=False)[
            ["beta_s_per_lap", "compound_offset_s", "fuel_effect_s_per_lap"]
        ].mean()
    deg, offset = {}, {}
    for k, g in part.groupby("compound"):
        g = g.sort_values("draw")
        deg[k] = g["beta_s_per_lap"].to_numpy()
        offset[k] = g["compound_offset_s"].to_numpy()
    fuel = part.groupby("draw")["fuel_effect_s_per_lap"].first().sort_index().to_numpy()
    for k in DRY:  # a compound never run at this circuit: neutral pace, average wear
        deg.setdefault(k, np.full(len(fuel), 0.05))
        offset.setdefault(k, np.zeros(len(fuel)))
    return deg, offset, fuel


def build_race_inputs(
    season: int,
    sessions: pd.DataFrame,
    results: pd.DataFrame,
    stints: pd.DataFrame,
    pit_losses: pd.DataFrame,
    sc_events: pd.DataFrame,
    posterior: pd.DataFrame,
    pace: PaceSettings | None = None,
) -> list[RaceInputs]:
    """`results` may span all seasons: earlier ones feed the retirement hazard."""
    pace = pace or PaceSettings()
    loss_by_circuit, loss_overall, loss_sd = pit_loss_by_circuit(pit_losses, season)
    hazard_by_circuit, hazard_global = sc_hazard_by_circuit(sc_events, season)
    dnf = dnf_hazard(results, season)
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
        deg, offset, fuel = posterior_draws(posterior, circuit)
        params = FieldParams(
            race_laps=laps,
            base_lap_s=float(pole * 1.07) if pd.notna(pole) else 95.0,
            deg_draws=deg,
            fuel_draws=fuel,
            compound_offset_draws=offset,
            pit_loss_mean_s=float(loss_by_circuit.get(circuit, loss_overall)),
            pit_loss_sd_s=loss_sd,
            sc_hazard_per_lap=float(hazard_by_circuit.get(circuit, hazard_global)),
            dnf_hazard_per_lap=dnf,
            pace_scale=pace.pace_scale,
            pace_noise_sd_s=pace.pace_noise_sd_s,
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


SCALAR_PARAMS = [
    "race_laps",
    "base_lap_s",
    "pit_loss_mean_s",
    "pit_loss_sd_s",
    "sc_hazard_per_lap",
    "dnf_hazard_per_lap",
    "pace_scale",
    "pace_noise_sd_s",
]


def to_tables(inputs: list[RaceInputs], season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persistable form: one row per race (scalars) and one per car (grid, pace, strategy).

    Posterior draws are not copied: readers join the RQ1 posterior by circuit.
    """
    races, cars = [], []
    for ri in inputs:
        p = ri.setup.params
        races.append(
            {
                "season": season,
                "race_id": ri.setup.race_id,
                "circuit_id": ri.circuit_id,
                **{name: getattr(p, name) for name in SCALAR_PARAMS},
            }
        )
        for i, car in enumerate(ri.setup.cars):
            cars.append(
                {
                    "season": season,
                    "race_id": ri.setup.race_id,
                    "car_index": i,
                    "driver_id": car.driver_id,
                    "grid": car.grid,
                    "pace_offset_s": car.pace_offset_s,
                    "actual_finish": int(ri.actual_finish[i]),
                    "strategy": ";".join(f"{s.compound}:{s.laps}" for s in car.strategy.stints),
                }
            )
    return pd.DataFrame(races), pd.DataFrame(cars)


def parse_strategy(text: str, name: str = "actual") -> Strategy:
    """'MEDIUM:20;HARD:37' -> Strategy."""
    stints = tuple(Stint(c, int(n)) for c, n in (part.split(":") for part in text.split(";")))
    return Strategy(name, stints)


def from_tables(race: pd.Series, cars: pd.DataFrame, posterior: pd.DataFrame) -> RaceSetup:
    """Rebuild a RaceSetup from the persisted tables (used by the app; no Spark)."""
    deg, offset, fuel = posterior_draws(posterior, race["circuit_id"])
    params = FieldParams(
        deg_draws=deg,
        fuel_draws=fuel,
        compound_offset_draws=offset,
        **{name: race[name] for name in SCALAR_PARAMS if name in race},
    )
    ordered = cars.sort_values("car_index")
    field_cars = tuple(
        Car(r.driver_id, int(r.grid), float(r.pace_offset_s), parse_strategy(r.strategy))
        for r in ordered.itertuples(index=False)
    )
    return RaceSetup(race["race_id"], field_cars, params)
