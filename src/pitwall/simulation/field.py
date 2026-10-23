"""RQ3: full-field Monte Carlo race simulator (docs/research-plan.md#rq3).

Lap by lap, vectorized over simulations (arrays are n_sims x n_cars):

- lap time = base + car pace + compound pace + fuel effect x laps left
  + degradation x tyre age + noise. Degradation, compound pace and fuel effect come from
  one RQ1 posterior draw per simulation, so parameter uncertainty propagates. Wear is
  floored at zero (a negative posterior draw is track evolution, not tyres getting faster);
- car pace = pace_scale x pre-race pace offset + a per-race deviation drawn once per car
  and simulation: qualifying gaps only partly predict race pace;
- pit stops at the end of each stint's last lap cost the pit loss, discounted under SC/VSC;
- safety cars start with a per-lap hazard, last a few laps, slow everyone and bunch the
  field up (order kept, ~1 s gaps);
- overtaking: a car that is faster on the road only gets past if its pace advantage clears
  a threshold, with a logistic chance; otherwise it is held up behind. Position changes
  through the pit lane are never blocked, which is what makes the undercut possible;
- retirements happen with a small per-lap hazard.

The output is the finishing order of every simulation, scored into points downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitwall.simulation.race_simulator import Strategy

POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)


@dataclass(frozen=True)
class Car:
    driver_id: str
    grid: int
    pace_offset_s: float  # per-lap pace relative to the field reference (lower = faster)
    strategy: Strategy


@dataclass(frozen=True)
class FieldParams:
    race_laps: int
    base_lap_s: float
    deg_draws: dict[str, np.ndarray]  # compound -> posterior draws of s/lap per lap of age
    fuel_draws: np.ndarray  # posterior draws of s/lap per remaining lap of fuel
    pit_loss_mean_s: float
    pit_loss_sd_s: float = 1.0
    sc_hazard_per_lap: float = 0.02
    sc_duration_laps: int = 4
    sc_lap_factor: float = 1.35  # SC lap time relative to the base lap
    sc_pit_discount: float = 0.45  # share of pit loss saved when stopping under SC
    sc_gap_s: float = 1.0  # gap between cars after the field bunches behind the SC
    lap_noise_sd_s: float = 0.35
    start_spacing_s: float = 0.4  # time per grid slot after the opening lap
    pass_threshold_s: float = 0.6  # pace advantage needed for even odds of passing
    pass_scale_s: float = 0.25
    min_follow_gap_s: float = 0.4
    dnf_hazard_per_lap: float = 0.0006
    pace_scale: float = 1.0  # multiplier on the pre-race pace offsets
    pace_noise_sd_s: float = 0.0  # per-car, per-simulation race pace deviation (s/lap)
    # compound -> draws of fresh-tyre pace (s), aligned draw-by-draw with deg_draws.
    compound_offset_draws: dict[str, np.ndarray] | None = None


def _stint_plan(strategy: Strategy, race_laps: int) -> tuple[list[str], np.ndarray]:
    """Compound per lap (1-based laps -> index lap-1) and the laps after which a car pits."""
    if strategy.total_laps != race_laps:
        raise ValueError(f"{strategy.name}: {strategy.total_laps} laps != {race_laps}")
    compounds: list[str] = []
    pit_after = []
    for stint in strategy.stints:
        compounds += [stint.compound] * stint.laps
        pit_after.append(len(compounds))
    return compounds, np.array(pit_after[:-1], dtype=int)


def simulate_field(
    cars: list[Car], params: FieldParams, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """Finishing positions (1 = winner), shape (n_sims, n_cars), in the order of `cars`."""
    n, laps = len(cars), params.race_laps
    plans = [_stint_plan(c.strategy, laps) for c in cars]
    pace = params.pace_scale * np.array([c.pace_offset_s for c in cars])[None, :]
    pace = pace + rng.normal(0, params.pace_noise_sd_s, (n_sims, n))

    # Parameter uncertainty: one joint posterior draw per simulation, shared by all cars.
    draw_idx = rng.integers(0, len(params.fuel_draws), n_sims)
    fuel = params.fuel_draws[draw_idx][:, None]
    deg = {k: np.maximum(v[draw_idx % len(v)], 0.0) for k, v in params.deg_draws.items()}
    offsets = params.compound_offset_draws or {}
    offset = {k: v[draw_idx % len(v)] for k, v in offsets.items()}
    no_offset = np.zeros(n_sims)

    grid_order = np.argsort(np.argsort([c.grid if c.grid > 0 else 99 for c in cars]))
    cum = np.tile(grid_order * params.start_spacing_s, (n_sims, 1)).astype(float)
    tyre_age = np.zeros((n_sims, n))
    alive = np.ones((n_sims, n), dtype=bool)
    laps_done = np.zeros((n_sims, n))
    sc_left = np.zeros(n_sims, dtype=int)

    for lap in range(1, laps + 1):
        starting_sc = (sc_left == 0) & (rng.random(n_sims) < params.sc_hazard_per_lap)
        sc_left = np.where(starting_sc, params.sc_duration_laps, sc_left)
        under_sc = sc_left > 0

        tyre_age += 1
        deg_now = np.empty((n_sims, n))
        offset_now = np.empty((n_sims, n))
        for i, (compounds, _) in enumerate(plans):
            deg_now[:, i] = deg[compounds[lap - 1]]
            offset_now[:, i] = offset.get(compounds[lap - 1], no_offset)
        lap_time = (
            params.base_lap_s
            + pace
            + offset_now
            + fuel * (laps - lap)
            + deg_now * tyre_age
            + rng.normal(0, params.lap_noise_sd_s, (n_sims, n))
        )
        order_before = np.argsort(np.where(alive, cum, np.inf), axis=1)
        tentative = cum + lap_time

        # Pit stops at the end of this lap.
        pitted = np.zeros((n_sims, n), dtype=bool)
        for i, (_, pit_after) in enumerate(plans):
            if lap in pit_after:
                pitted[:, i] = True
        # Drawn every lap (even with no stop) so the random stream doesn't depend on the
        # strategies: common random numbers across candidate strategies stay aligned.
        loss = rng.normal(params.pit_loss_mean_s, params.pit_loss_sd_s, (n_sims, n))
        loss = np.where(under_sc[:, None], loss * (1 - params.sc_pit_discount), loss)
        tentative = tentative + np.where(pitted, loss, 0.0)
        tyre_age = np.where(pitted, 0.0, tyre_age)

        if under_sc.any():
            # Field bunches behind the SC: keep the running order, fixed small gaps.
            order = np.argsort(np.where(alive, tentative, np.inf), axis=1)
            rank = np.argsort(order, axis=1)
            leader = np.take_along_axis(tentative, order[:, :1], axis=1)
            bunched = (
                leader + params.base_lap_s * (params.sc_lap_factor - 1) + rank * params.sc_gap_s
            )
            tentative = np.where(under_sc[:, None], bunched, tentative)
        else:
            tentative = _resolve_overtakes(tentative, cum, order_before, pitted, alive, params, rng)

        cum = np.where(alive, tentative, cum)
        laps_done += alive
        retire = alive & (rng.random((n_sims, n)) < params.dnf_hazard_per_lap)
        alive &= ~retire
        sc_left = np.maximum(sc_left - 1, 0)

    # Classified by laps completed (desc), then time.
    key = np.where(alive, cum, np.inf) - laps_done * 1e6
    return np.argsort(np.argsort(key, axis=1), axis=1) + 1


def _resolve_overtakes(
    tentative: np.ndarray,
    cum_before: np.ndarray,
    order_before: np.ndarray,
    pitted: np.ndarray,
    alive: np.ndarray,
    params: FieldParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Walk the running order front to back; a car that would pass the car ahead on track
    only does so if it wins a pace-dependent coin flip, else it is held at a following gap."""
    sims = np.arange(tentative.shape[0])
    out = tentative.copy()
    for pos in range(1, tentative.shape[1]):
        behind = order_before[:, pos]
        ahead = order_before[:, pos - 1]
        t_b, t_a = out[sims, behind], out[sims, ahead]
        on_track = (
            ~pitted[sims, behind] & ~pitted[sims, ahead] & alive[sims, behind] & alive[sims, ahead]
        )
        would_pass = on_track & (t_b < t_a + params.min_follow_gap_s)
        lap_b = t_b - cum_before[sims, behind]
        lap_a = t_a - cum_before[sims, ahead]
        advantage = lap_a - lap_b
        p_pass = 1 / (1 + np.exp(-(advantage - params.pass_threshold_s) / params.pass_scale_s))
        passes = would_pass & (rng.random(len(sims)) < p_pass)
        held = would_pass & ~passes
        out[sims[held], behind[held]] = t_a[held] + params.min_follow_gap_s
    return out


def points_for(positions: np.ndarray) -> np.ndarray:
    pts = np.zeros(positions.shape)
    top = positions <= len(POINTS)
    pts[top] = POINTS[positions[top] - 1]
    return pts
