"""RQ1: Bayesian hierarchical tire degradation model (docs/research-plan.md#rq1).

Input: feat_tire_degradation as pandas, with columns
lap_time_s, tyre_life_laps, compound, fuel_proxy, track_temp_c, in_traffic,
circuit_id, driver_id, race_id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import arviz as az
    import pymc as pm


def _standardize(x: pd.Series) -> np.ndarray:
    values = x.to_numpy(float)
    return (values - values.mean()) / (values.std() or 1.0)


def build_model(df: pd.DataFrame) -> pm.Model:
    import pymc as pm

    race_idx, races = pd.factorize(df["race_id"])
    driver_idx, drivers = pd.factorize(df["driver_id"])
    circuit_idx, circuits = pd.factorize(df["circuit_id"])
    compound_idx, compounds = pd.factorize(df["compound"])

    # Center lap times per race so race_offset captures only residual race-level pace.
    y = (df["lap_time_s"] - df.groupby("race_id")["lap_time_s"].transform("median")).to_numpy(float)
    tyre = df["tyre_life_laps"].to_numpy(float)
    fuel = df["fuel_proxy"].to_numpy(float)
    temp = _standardize(df["track_temp_c"])
    traffic = df["in_traffic"].to_numpy(float)

    coords = {"race": races, "driver": drivers, "circuit": circuits, "compound": compounds}
    with pm.Model(coords=coords) as model:
        race_offset = pm.Normal("race_offset", 0.0, 2.0, dims="race")
        driver_sigma = pm.HalfNormal("driver_sigma", 0.5)
        driver_offset = pm.Normal("driver_offset", 0.0, driver_sigma, dims="driver")

        # Seconds per lap of remaining fuel (heavier car is slower).
        fuel_effect = pm.Normal("fuel_effect", 0.04, 0.02)

        # Degradation slope: compound mean + circuit deviation (non-centered partial pooling).
        beta_compound = pm.Normal("beta_compound", 0.05, 0.1, dims="compound")
        circuit_sigma = pm.HalfNormal("circuit_sigma", 0.05)
        z = pm.Normal("z_circuit_compound", 0.0, 1.0, dims=("circuit", "compound"))
        beta = pm.Deterministic(
            "beta", beta_compound[None, :] + circuit_sigma * z, dims=("circuit", "compound")
        )
        gamma_temp = pm.Normal("gamma_temp", 0.0, 0.02)
        delta_traffic = pm.Normal("delta_traffic", 0.3, 0.3)

        mu = (
            race_offset[race_idx]
            + driver_offset[driver_idx]
            + fuel_effect * fuel
            + (beta[circuit_idx, compound_idx] + gamma_temp * temp) * tyre
            + delta_traffic * traffic
        )
        sigma = pm.HalfNormal("sigma", 1.0)
        nu = pm.Gamma("nu", 2.0, 0.1)  # Student-t: robust to residual outlier laps
        pm.StudentT("lap_time", nu=nu, mu=mu, sigma=sigma, observed=y)
    # TODO(phase 3): tyre "cliff" via piecewise/spline term; era factor (pre/post 2022).
    return model


def fit(
    df: pd.DataFrame, draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 42
) -> az.InferenceData:
    import pymc as pm

    with build_model(df):
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=seed)
    return idata
