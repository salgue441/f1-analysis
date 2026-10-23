"""RQ3: Monte Carlo race strategy simulator (docs/research-plan.md#rq3).

Strategy types and a single-car race-time simulator (no traffic), kept as the simplest
reference model. The full-field simulator with overtaking and safety cars is
`simulation.field`; Spark distribution is `simulation.distributed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stint:
    compound: str
    laps: int


@dataclass(frozen=True)
class Strategy:
    name: str
    stints: tuple[Stint, ...]

    @property
    def total_laps(self) -> int:
        return sum(s.laps for s in self.stints)


@dataclass(frozen=True)
class RaceParams:
    race_laps: int
    base_lap_s: float
    fuel_effect_s: float  # seconds per lap of remaining fuel
    deg_samples: dict[str, np.ndarray]  # compound -> posterior draws of s/lap (from RQ1)
    pit_loss_mean_s: float
    pit_loss_sd_s: float
    sc_hazard_per_lap: float  # P(safety car starts on a given lap)
    sc_pit_discount: float = 0.5  # fraction of pit loss saved under SC


def simulate_strategy(
    strategy: Strategy, params: RaceParams, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """Return an array of `n_sims` simulated total race times (seconds)."""
    if strategy.total_laps != params.race_laps:
        raise ValueError(f"{strategy.name}: {strategy.total_laps} laps != {params.race_laps}")

    laps = np.arange(1, params.race_laps + 1)
    fuel_time = params.fuel_effect_s * (params.race_laps - laps)  # same for every sim
    total = np.full(n_sims, params.base_lap_s * params.race_laps + fuel_time.sum())

    sc_laps = rng.random((n_sims, params.race_laps)) < params.sc_hazard_per_lap
    lap_cursor = 0
    for i, stint in enumerate(strategy.stints):
        draws = params.deg_samples[stint.compound]
        deg = rng.choice(draws, size=n_sims)  # parameter uncertainty
        tyre_age = np.arange(1, stint.laps + 1)
        total += deg * tyre_age.sum()
        lap_cursor += stint.laps
        if i < len(strategy.stints) - 1:  # pit at end of stint
            pit_loss = rng.normal(params.pit_loss_mean_s, params.pit_loss_sd_s, n_sims)
            under_sc = sc_laps[:, lap_cursor - 1]
            total += np.where(under_sc, pit_loss * (1 - params.sc_pit_discount), pit_loss)
    return total
