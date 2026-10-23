"""Simulated feat_tire_degradation data with known parameters (research plan: recovery test).

Generated from the same structure the RQ1 model assumes, so fitting it should recover
`truth` within the posterior intervals. Used by the slow test suite and the RQ1 report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass(frozen=True)
class Truth:
    beta_compound: dict[str, float] = field(
        default_factory=lambda: {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.025}
    )
    # Fresh-tyre pace (s), zero-sum over compounds like the model's.
    alpha_compound: dict[str, float] = field(
        default_factory=lambda: {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.5}
    )
    circuit_sigma: float = 0.02
    driver_deg_sigma: float = 0.005
    fuel_effect: float = 0.035
    gamma_temp: float = 0.006
    delta_traffic: float = 0.4
    sigma: float = 0.35
    nu: float = 5.0


def simulate_laps(
    n_circuits: int = 8,
    races_per_circuit: int = 2,
    n_drivers: int = 12,
    race_laps: int = 50,
    truth: Truth | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Return (laps, true circuit x compound betas). Two stints per driver per race."""
    truth = truth or Truth()
    rng = np.random.default_rng(seed)
    circuits = [f"c{i}" for i in range(n_circuits)]
    drivers = [f"d{i}" for i in range(n_drivers)]
    beta = {
        (c, k): truth.beta_compound[k] + truth.circuit_sigma * rng.standard_normal()
        for c in circuits
        for k in COMPOUNDS
    }
    driver_pace = dict(zip(drivers, rng.normal(0, 0.3, n_drivers), strict=True))
    driver_deg = dict(zip(drivers, rng.normal(0, truth.driver_deg_sigma, n_drivers), strict=True))

    rows = []
    for c in circuits:
        for r in range(races_per_circuit):
            race_id = f"{c}_r{r}"
            race_pace = rng.normal(90, 5)
            base_temp = rng.uniform(20, 50)
            for d in drivers:
                pit = int(rng.integers(15, 35))
                first, second = rng.choice(COMPOUNDS, size=2, replace=False)
                for stint_number, (compound, start, end) in enumerate(
                    [(first, 2, pit), (second, pit + 1, race_laps)], start=1
                ):
                    for lap in range(start, end + 1):
                        tyre = lap - start + 1
                        temp = base_temp + rng.normal(0, 1.5)
                        traffic = rng.random() < 0.2
                        fuel = race_laps - lap
                        mu = (
                            race_pace
                            + driver_pace[d]
                            + truth.alpha_compound[compound]
                            + truth.fuel_effect * fuel
                            + (
                                beta[(c, compound)]
                                + driver_deg[d]
                                + truth.gamma_temp * (temp - 35) / 9
                            )
                            * tyre
                            + truth.delta_traffic * traffic
                        )
                        y = mu + truth.sigma * rng.standard_t(truth.nu)
                        rows.append(
                            (
                                race_id,
                                c,
                                d,
                                stint_number,
                                lap,
                                compound,
                                tyre,
                                fuel,
                                temp,
                                traffic,
                                y,
                            )
                        )
    df = pd.DataFrame(
        rows,
        columns=[
            "race_id",
            "circuit_id",
            "driver_id",
            "stint_number",
            "lap_number",
            "compound",
            "tyre_life_laps",
            "fuel_proxy",
            "track_temp_c",
            "in_traffic",
            "lap_time_s",
        ],
    )
    return df, beta
