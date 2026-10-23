"""Candidate strategies and per-driver strategy evaluation for RQ3."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from pitwall.simulation.field import Car, FieldParams, points_for, simulate_field
from pitwall.simulation.race_simulator import Stint, Strategy

DRY = ("SOFT", "MEDIUM", "HARD")


def candidate_strategies(
    race_laps: int, min_stint: int = 8, step: int = 2, max_stops: int = 2
) -> list[Strategy]:
    """Legal dry strategies (>= 2 compounds) with 1 or 2 stops on a lap grid."""
    out: list[Strategy] = []
    for c1, c2 in itertools.permutations(DRY, 2):
        for pit in range(min_stint, race_laps - min_stint + 1, step):
            out.append(
                Strategy(f"{c1[0]}{c2[0]}-{pit}", (Stint(c1, pit), Stint(c2, race_laps - pit)))
            )
    if max_stops >= 2:
        step2 = step * 2
        for c1, c2, c3 in itertools.product(DRY, repeat=3):
            if len({c1, c2, c3}) < 2:
                continue
            for p1 in range(min_stint, race_laps - 2 * min_stint + 1, step2):
                for p2 in range(p1 + min_stint, race_laps - min_stint + 1, step2):
                    stints = (Stint(c1, p1), Stint(c2, p2 - p1), Stint(c3, race_laps - p2))
                    out.append(Strategy(f"{c1[0]}{c2[0]}{c3[0]}-{p1}-{p2}", stints))
    return out


def strategy_from_stints(stints: pd.DataFrame, race_laps: int, name: str = "actual") -> Strategy:
    """Turn fact_stint rows (compound, laps, ordered) into a Strategy covering `race_laps`.

    Wet or unknown stints are mapped to MEDIUM (dry-race simulator), and the last stint is
    stretched or trimmed so the total matches the scheduled distance.
    """
    rows = stints.sort_values("stint_number")
    plan = [
        Stint(c if c in DRY else "MEDIUM", int(n))
        for c, n in zip(rows["compound"], rows["laps"], strict=True)
        if n > 0
    ]
    if not plan:
        plan = [Stint("MEDIUM", race_laps)]
    total = sum(s.laps for s in plan)
    last = plan[-1]
    plan[-1] = Stint(last.compound, max(1, last.laps + race_laps - total))
    while sum(s.laps for s in plan) > race_laps:  # trimming could need earlier stints
        plan.pop()
        plan[-1] = Stint(plan[-1].compound, race_laps - sum(s.laps for s in plan[:-1]))
    return Strategy(name, tuple(plan))


def evaluate_for_driver(
    cars: list[Car],
    target: int,
    candidates: list[Strategy],
    params: FieldParams,
    n_sims: int,
    seed: int,
) -> pd.DataFrame:
    """Expected points and finish of car `target` under each candidate strategy, everyone
    else fixed. Common random numbers: every candidate sees the same seed, so differences
    come from the strategy, not from luck."""
    actual = cars[target].strategy
    base_pts = points_for(
        simulate_field(cars, params, n_sims, np.random.default_rng(seed))[:, target]
    )
    rows = []
    for index, strategy in enumerate([actual, *candidates]):
        trial = list(cars)
        trial[target] = replace(cars[target], strategy=strategy)
        pos = simulate_field(trial, params, n_sims, np.random.default_rng(seed))[:, target]
        pts = points_for(pos)
        rows.append(
            {
                "strategy": strategy.name,
                "stops": len(strategy.stints) - 1,
                "expected_points": pts.mean(),
                "expected_position": pos.mean(),
                "p_more_points_than_actual": float((pts > base_pts).mean()),
                "p_fewer_points_than_actual": float((pts < base_pts).mean()),
                "points_gain_vs_actual": float((pts - base_pts).mean()),
                "points_gain_se": float((pts - base_pts).std(ddof=1) / np.sqrt(n_sims)),
                "is_actual": index == 0,
            }
        )
    return pd.DataFrame(rows)
