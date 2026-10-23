"""RQ1: Bayesian hierarchical tire degradation model (docs/research-plan.md#rq1).

Input: feat_tire_degradation as pandas, with columns
lap_time_s, tyre_life_laps, compound, fuel_proxy, track_temp_c, in_traffic,
circuit_id, driver_id, race_id.

Model (lap time centred on the race median):
    y = race_offset[race] + driver_offset[driver] + alpha[circuit, compound]
        + fuel_effect * fuel
        + (beta[circuit, compound] + driver_deg[driver] + gamma_temp * temp_std) * tyre_life
        + delta_traffic * in_traffic + StudentT noise
with beta[c, k] = beta_compound[k] + circuit_sigma * z[c, k] (non-centred partial pooling)
and alpha[c, k] the fresh-tyre pace of compound k at circuit c, pooled the same way and
summing to zero over compounds (the overall level belongs to the race offset).

Forecasting a new stint only needs the *shape* terms (fuel, tyre, temperature, traffic):
race and driver offsets are constant within a stint, so they cancel when the forecast is
anchored on the stint's first laps (`models.evaluation`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pitwall.models.evaluation import StintSplit, anchored

if TYPE_CHECKING:
    import arviz as az
    import pymc as pm

SHAPE_VARS = [
    "fuel_effect",
    "alpha_compound",
    "alpha_circuit_sigma",
    "alpha",
    "beta_compound",
    "circuit_sigma",
    "beta",
    "driver_deg_sigma",
    "driver_deg",
    "gamma_temp",
    "delta_traffic",
    "sigma",
    "nu",
]


@dataclass(frozen=True)
class TempScaler:
    """Track temperature standardization fitted on the training set only."""

    mean: float
    sd: float

    @classmethod
    def fit(cls, temps: pd.Series) -> TempScaler:
        values = temps.to_numpy(float)
        return cls(float(values.mean()), float(values.std() or 1.0))

    def transform(self, temps: pd.Series) -> np.ndarray:
        return (temps.to_numpy(float) - self.mean) / self.sd


def center_by_race(df: pd.DataFrame) -> np.ndarray:
    return (df["lap_time_s"] - df.groupby("race_id")["lap_time_s"].transform("median")).to_numpy(
        float
    )


def build_model(
    df: pd.DataFrame,
    scaler: TempScaler | None = None,
    fuel_prior: tuple[float, float] = (0.04, 0.02),
) -> pm.Model:
    import pymc as pm

    scaler = scaler or TempScaler.fit(df["track_temp_c"])
    race_idx, races = pd.factorize(df["race_id"])
    driver_idx, drivers = pd.factorize(df["driver_id"])
    circuit_idx, circuits = pd.factorize(df["circuit_id"])
    compound_idx, compounds = pd.factorize(df["compound"])

    y = center_by_race(df)
    tyre = df["tyre_life_laps"].to_numpy(float)
    fuel = df["fuel_proxy"].to_numpy(float)
    temp = scaler.transform(df["track_temp_c"])
    traffic = df["in_traffic"].to_numpy(float)

    coords = {"race": races, "driver": drivers, "circuit": circuits, "compound": compounds}
    with pm.Model(coords=coords) as model:
        race_offset = pm.Normal("race_offset", 0.0, 2.0, dims="race")
        driver_sigma = pm.HalfNormal("driver_sigma", 0.5)
        # Sum-to-zero: race offsets already absorb each race's average pace, so an
        # unconstrained driver mean is barely identified and driver_sigma mixes slowly.
        driver_offset = pm.ZeroSumNormal("driver_offset", sigma=driver_sigma, dims="driver")

        # Seconds per lap of remaining fuel (heavier car is slower).
        fuel_effect = pm.Normal("fuel_effect", *fuel_prior)

        # Degradation slope: compound mean + circuit deviation (non-centered partial pooling).
        beta_compound = pm.Normal("beta_compound", 0.05, 0.1, dims="compound")
        circuit_sigma = pm.HalfNormal("circuit_sigma", 0.05)
        z = pm.Normal("z_circuit_compound", 0.0, 1.0, dims=("circuit", "compound"))
        beta = pm.Deterministic(
            "beta", beta_compound[None, :] + circuit_sigma * z, dims=("circuit", "compound")
        )
        # Driver deviation in degradation (tyre management), pooled: tests H1c against
        # circuit_sigma on the same scale (s/lap per lap of tyre age).
        driver_deg_sigma = pm.HalfNormal("driver_deg_sigma", 0.02)
        z_driver = pm.Normal("z_driver_deg", 0.0, 1.0, dims="driver")
        driver_deg = pm.Deterministic("driver_deg", driver_deg_sigma * z_driver, dims="driver")
        # Fresh-tyre pace by compound (s): soft is quicker when new, hard slower. Zero-sum over
        # compounds, since the overall level is the race offset's.
        alpha_compound = pm.ZeroSumNormal("alpha_compound", sigma=1.0, dims="compound")
        alpha_circuit_sigma = pm.HalfNormal("alpha_circuit_sigma", 0.5)
        z_alpha = pm.ZeroSumNormal(
            "z_alpha", sigma=1.0, dims=("circuit", "compound"), n_zerosum_axes=1
        )
        alpha = pm.Deterministic(
            "alpha",
            alpha_compound[None, :] + alpha_circuit_sigma * z_alpha,
            dims=("circuit", "compound"),
        )
        gamma_temp = pm.Normal("gamma_temp", 0.0, 0.02)
        delta_traffic = pm.Normal("delta_traffic", 0.3, 0.3)

        mu = (
            race_offset[race_idx]
            + driver_offset[driver_idx]
            + alpha[circuit_idx, compound_idx]
            + fuel_effect * fuel
            + (beta[circuit_idx, compound_idx] + driver_deg[driver_idx] + gamma_temp * temp) * tyre
            + delta_traffic * traffic
        )
        sigma = pm.HalfNormal("sigma", 1.0)
        nu = pm.Gamma("nu", 2.0, 0.1)  # Student-t: robust to residual outlier laps
        pm.StudentT("lap_time", nu=nu, mu=mu, sigma=sigma, observed=y)
    model.temp_scaler = scaler  # type: ignore[attr-defined]
    return model


def fit(
    df: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 42,
    sampler: str = "auto",
    fuel_prior: tuple[float, float] = (0.04, 0.02),
) -> tuple[az.InferenceData, TempScaler]:
    """NUTS fit. `sampler="auto"` uses nutpie when installed (much faster), else PyMC."""
    import pymc as pm

    scaler = TempScaler.fit(df["track_temp_c"])
    model = build_model(df, scaler, fuel_prior)
    if sampler == "auto":
        try:
            import nutpie  # noqa: F401

            sampler = "nutpie"
        except ImportError:
            sampler = "pymc"
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            nuts_sampler=sampler,
            progressbar=False,
        )
    return idata, scaler


def diagnostics(idata: az.InferenceData) -> dict[str, float]:
    """Convergence summary for the research plan's thresholds (R-hat, ESS, divergences)."""
    import arviz as az

    names = [*SHAPE_VARS, "driver_sigma"]
    summary = az.summary(idata, var_names=names, kind="diagnostics")
    divergences = (
        int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
    )
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "divergences": divergences,
        "worst_ess": summary["ess_bulk"].nsmallest(3).round(0).to_dict(),
        "worst_rhat": summary["r_hat"].nlargest(3).round(3).to_dict(),
    }


def posterior_draws(idata: az.InferenceData, n_draws: int | None = None, seed: int = 0) -> dict:
    """Flatten chains x draws for the shape parameters, optionally subsampled."""
    post = idata.posterior.stack(sample=("chain", "draw"))
    total = post.sizes["sample"]
    rng = np.random.default_rng(seed)
    pick = (
        rng.choice(total, size=n_draws, replace=False)
        if n_draws and n_draws < total
        else slice(None)
    )
    out = {}
    for var in SHAPE_VARS:
        values = post[var].values  # sample axis last
        out[var] = np.moveaxis(values, -1, 0)[pick]
    out["circuits"] = list(post.coords["circuit"].values)
    out["compounds"] = list(post.coords["compound"].values)
    out["drivers"] = list(post.coords["driver"].values)
    return out


def shape_draws(draws: dict, df: pd.DataFrame, scaler: TempScaler, seed: int = 0) -> np.ndarray:
    """Posterior draws of the within-stint shape terms for each row: (n_draws, n_rows).

    Unseen circuits get a fresh circuit deviation from the hierarchical prior per draw;
    an unseen compound raises (nothing to pool from).
    """
    rng = np.random.default_rng(seed)
    n = draws["fuel_effect"].shape[0]
    circuit_pos = {c: i for i, c in enumerate(draws["circuits"])}
    compound_pos = {k: i for i, k in enumerate(draws["compounds"])}
    unknown = set(df["compound"]) - set(compound_pos)
    if unknown:
        raise ValueError(f"Compounds not in the fitted model: {sorted(unknown)}")

    k_idx = df["compound"].map(compound_pos).to_numpy()
    beta = np.empty((n, len(df)))
    seen = df["circuit_id"].isin(circuit_pos).to_numpy()
    if seen.any():
        c_idx = df.loc[seen, "circuit_id"].map(circuit_pos).to_numpy()
        beta[:, seen] = draws["beta"][:, c_idx, k_idx[seen]]
    for circuit in df.loc[~seen, "circuit_id"].unique():
        rows = (df["circuit_id"] == circuit).to_numpy()
        z_new = rng.standard_normal((n, len(compound_pos)))
        beta_new = draws["beta_compound"] + draws["circuit_sigma"][:, None] * z_new
        beta[:, rows] = beta_new[:, k_idx[rows]]

    driver_pos = {d: i for i, d in enumerate(draws["drivers"])}
    d_idx = df["driver_id"].map(driver_pos)
    known = d_idx.notna().to_numpy()
    driver_deg = np.empty((n, len(df)))
    driver_deg[:, known] = draws["driver_deg"][:, d_idx[known].astype(int).to_numpy()]
    # One fresh deviation per unseen driver and draw, shared by all of that driver's rows.
    for driver in df.loc[~known, "driver_id"].unique():
        rows = (df["driver_id"] == driver).to_numpy()
        driver_deg[:, rows] = (draws["driver_deg_sigma"] * rng.standard_normal(n))[:, None]
    beta = beta + driver_deg

    temp = scaler.transform(df["track_temp_c"])[None, :]
    tyre = df["tyre_life_laps"].to_numpy(float)[None, :]
    fuel = df["fuel_proxy"].to_numpy(float)[None, :]
    traffic = df["in_traffic"].to_numpy(float)[None, :]
    return (
        draws["fuel_effect"][:, None] * fuel
        + (beta + draws["gamma_temp"][:, None] * temp) * tyre
        + draws["delta_traffic"][:, None] * traffic
    )


def degradation_table(draws: dict, interval: float = 0.9) -> pd.DataFrame:
    """Posterior mean and interval of beta (s/lap of tyre age) per circuit x compound."""
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    rows = []
    for i, circuit in enumerate(draws["circuits"]):
        for j, compound in enumerate(draws["compounds"]):
            b = draws["beta"][:, i, j]
            rows.append((circuit, compound, b.mean(), np.quantile(b, lo), np.quantile(b, hi)))
    for j, compound in enumerate(draws["compounds"]):
        b = draws["beta_compound"][:, j]
        rows.append(("(all circuits)", compound, b.mean(), np.quantile(b, lo), np.quantile(b, hi)))
    return pd.DataFrame(rows, columns=["circuit_id", "compound", "mean", "lo", "hi"])


def export_posterior(draws: dict) -> pd.DataFrame:
    """Long table of posterior draws for the RQ3 simulator: circuit x compound x draw.

    `compound_offset_s` is the fresh-tyre pace of the compound at that circuit relative to
    the circuit's compound average (negative = faster).
    """
    n = draws["beta"].shape[0]
    frames = []
    for i, circuit in enumerate(draws["circuits"]):
        for j, compound in enumerate(draws["compounds"]):
            frames.append(
                pd.DataFrame(
                    {
                        "circuit_id": circuit,
                        "compound": compound,
                        "draw": np.arange(n),
                        "beta_s_per_lap": draws["beta"][:, i, j],
                        "compound_offset_s": draws["alpha"][:, i, j],
                        "fuel_effect_s_per_lap": draws["fuel_effect"],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def predictive_draws(
    draws: dict, split: StintSplit, scaler: TempScaler, seed: int = 0
) -> np.ndarray:
    """Posterior predictive forecasts of the target laps.

    Lap noise is simulated on the anchor laps too: the observed anchor mean is itself noisy,
    so the level it sets is uncertain (variance ~ sigma^2 / n_anchor), and leaving that out
    makes intervals too narrow. Anchor and target rows are drawn together, so an unseen
    driver's (or circuit's) fresh deviation is the same in both.
    """
    both = pd.concat([split.anchor, split.target], ignore_index=True)
    shape = shape_draws(draws, both, scaler, seed)
    rng = np.random.default_rng(seed + 1)
    noise = rng.standard_t(draws["nu"][:, None], size=shape.shape) * draws["sigma"][:, None]
    shape = shape + noise
    n_anchor = len(split.anchor)
    return anchored(split, shape[:, :n_anchor], shape[:, n_anchor:])


def hypothesis_tests(draws: dict) -> dict[str, float]:
    """Posterior probabilities for H1a-H1c (research plan §RQ1)."""
    k = {c: i for i, c in enumerate(draws["compounds"])}
    out: dict[str, float] = {}
    if {"SOFT", "MEDIUM", "HARD"} <= set(k):
        b = draws["beta"]
        ordered = (b[:, :, k["SOFT"]] > b[:, :, k["MEDIUM"]]) & (
            b[:, :, k["MEDIUM"]] > b[:, :, k["HARD"]]
        )
        p_by_circuit = ordered.mean(axis=0)
        out["h1a_mean_p_ordered"] = float(p_by_circuit.mean())
        out["h1a_share_circuits_p_gt_half"] = float((p_by_circuit > 0.5).mean())
        bc = draws["beta_compound"]
        out["h1a_p_ordered_overall"] = float(
            (
                (bc[:, k["SOFT"]] > bc[:, k["MEDIUM"]]) & (bc[:, k["MEDIUM"]] > bc[:, k["HARD"]])
            ).mean()
        )
    out["h1b_p_gamma_temp_positive"] = float((draws["gamma_temp"] > 0).mean())
    out["h1c_p_circuit_sd_gt_driver_sd"] = float(
        (draws["circuit_sigma"] > draws["driver_deg_sigma"]).mean()
    )
    return out


def pace_offset_table(draws: dict, interval: float = 0.9) -> pd.DataFrame:
    """Fresh-tyre pace by compound (s, relative to the compound average): overall."""
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    a = draws["alpha_compound"]
    return pd.DataFrame(
        [
            (k, a[:, j].mean(), np.quantile(a[:, j], lo), np.quantile(a[:, j], hi))
            for j, k in enumerate(draws["compounds"])
        ],
        columns=["compound", "mean", "lo", "hi"],
    )
