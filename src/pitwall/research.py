"""Feature builds and research-question runs (the science side of the I/O wiring).

`pipelines` builds the lakehouse; this module turns gold into feature tables and runs each
research question end to end: baselines, main model, diagnostics, robustness checks, MLflow
logging, and the small result files the reports in `reports/` are written from.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from pitwall.features.tire_degradation import build_feat_tire_degradation
from pitwall.gold.writer import overwrite_season
from pitwall.paths import table_path
from pitwall.spark import get_spark

log = logging.getLogger(__name__)

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def _read(layer: str, table: str):
    return get_spark().read.format("delta").load(table_path(layer, table))  # type: ignore[arg-type]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")


# --- RQ1 -------------------------------------------------------------------------------------


def build_feat_rq1(season: int) -> int:
    spark = get_spark()
    in_season = F.col("season") == season
    feat = build_feat_tire_degradation(
        _read("gold", "fact_lap").filter(in_season),
        _read("gold", "dim_session").filter(in_season),
        _read("gold", "fact_session_result").filter(in_season),
    )
    path = table_path("features", "feat_tire_degradation")
    overwrite_season(spark, feat, path, season)
    return _read("features", "feat_tire_degradation").filter(in_season).count()


def load_feat_rq1(seasons: list[int]) -> pd.DataFrame:
    df = _read("features", "feat_tire_degradation").filter(F.col("season").isin(seasons))
    return df.toPandas()


def _forest_plot(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"SOFT": "#d62728", "MEDIUM": "#e6b800", "HARD": "#7f7f7f"}
    circuits = sorted(table["circuit_id"].unique(), key=lambda c: (c == "(all circuits)", c))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(circuits) + 1.5))
    for j, compound in enumerate(["SOFT", "MEDIUM", "HARD"]):
        part = table[table["compound"] == compound].set_index("circuit_id").reindex(circuits)
        y = np.arange(len(circuits)) + (j - 1) * 0.25
        ax.errorbar(
            part["mean"],
            y,
            xerr=[part["mean"] - part["lo"], part["hi"] - part["mean"]],
            fmt="o",
            ms=3.5,
            color=colors[compound],
            label=compound,
            capsize=0,
            lw=1,
        )
    ax.set_yticks(np.arange(len(circuits)), circuits)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Degradation (s per lap of tyre age), posterior mean and 90% interval")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ppc_plot(y: np.ndarray, pred: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PIT histogram: a calibrated forecast gives a flat histogram.
    pit = (pred < y[None, :]).mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(pit, bins=20, color="#4c72b0", edgecolor="white")
    axes[0].axhline(len(pit) / 20, color="black", lw=0.8, ls="--")
    axes[0].set_title("PIT histogram, 2025 held-out laps")
    axes[0].set_xlabel("Predictive CDF at observed lap time")
    resid = y - np.median(pred, axis=0)
    axes[1].hist(np.clip(resid, -4, 4), bins=60, color="#55a868", edgecolor="white")
    axes[1].set_title("Forecast error (observed - median), clipped at ±4 s")
    axes[1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_rq1(
    train_seasons: list[int],
    test_season: int,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 42,
    sensitivity: bool = True,
) -> dict[str, Any]:
    """Research plan §RQ1 end to end. Writes reports/rq1/*.json|csv|png and an MLflow run."""
    import mlflow

    from pitwall.models import baselines, evaluation
    from pitwall.models import tire_degradation as td
    from pitwall.tracking import tracked_run

    out = REPORTS / "rq1"
    train = load_feat_rq1(train_seasons)
    test = load_feat_rq1([test_season])
    split = evaluation.split_stints(test)
    y = split.target["lap_time_s"].to_numpy()
    log.info("RQ1: %s train laps, %s test laps (%s scored)", len(train), len(test), len(y))

    params = {
        "train_seasons": ",".join(map(str, train_seasons)),
        "test_season": test_season,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "n_train": len(train),
        "n_test_scored": len(y),
    }
    tables = {"feat_tire_degradation": table_path("features", "feat_tire_degradation")}
    with tracked_run("rq1-tire-degradation", params, tables, seed) as run:
        results: dict[str, Any] = {"mlflow_run_id": run.info.run_id, **params}

        b0 = baselines.B0CompoundSlope().fit(train)
        results["b0"] = evaluation.score(y, *b0.predict(split))
        results["b0_slopes"] = b0.slopes
        b1 = baselines.B1LightGBM(seed=seed).fit(train)
        results["b1"] = evaluation.score(y, *b1.predict(split))

        idata, scaler = td.fit(train, draws=draws, tune=tune, chains=chains, seed=seed)
        results["diagnostics"] = td.diagnostics(idata)
        post = td.posterior_draws(idata)
        pred = td.predictive_draws(
            td.posterior_draws(idata, n_draws=1000, seed=seed), split, scaler, seed
        )
        results["bayes"] = evaluation.score_draws(y, pred)
        results["hypotheses"] = td.hypothesis_tests(post)
        results["global"] = {
            name: {
                "mean": float(post[name].mean()),
                "lo": float(np.quantile(post[name], 0.05)),
                "hi": float(np.quantile(post[name], 0.95)),
            }
            for name in [
                "fuel_effect",
                "gamma_temp",
                "delta_traffic",
                "circuit_sigma",
                "driver_deg_sigma",
                "sigma",
                "nu",
            ]
        }
        results["beta_compound"] = {
            k: {
                "mean": float(post["beta_compound"][:, j].mean()),
                "lo": float(np.quantile(post["beta_compound"][:, j], 0.05)),
                "hi": float(np.quantile(post["beta_compound"][:, j], 0.95)),
            }
            for j, k in enumerate(post["compounds"])
        }
        # Coverage by compound: is the interval honest everywhere, not just on average?
        results["bayes_by_compound"] = {
            k: evaluation.score_draws(y[mask], pred[:, mask])
            for k in split.target["compound"].unique()
            if (mask := (split.target["compound"] == k).to_numpy()).any()
        }

        table = td.degradation_table(post)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "degradation_table.csv", index=False, float_format="%.4f")
        _forest_plot(table, out / "forest_plot.png")
        _ppc_plot(y, pred, out / "calibration.png")

        if sensitivity:
            results["sensitivity"] = _rq1_sensitivity(
                train, seed, draws=max(draws // 2, 300), tune=max(tune // 2, 300), chains=chains
            )

        export = td.export_posterior(td.posterior_draws(idata, n_draws=1000, seed=seed))
        get_spark().createDataFrame(export).write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(table_path("features", "rq1_degradation_posterior"))

        for name in ["b0", "b1", "bayes"]:
            mlflow.log_metrics({f"{name}_{k}": v for k, v in results[name].items()})
        mlflow.log_metrics({k: float(v) for k, v in results["diagnostics"].items()})
        mlflow.log_metrics(results["hypotheses"])
        mlflow.log_artifacts(str(out), "rq1")
        _write_json(out / "results.json", results)
    return results


def _rq1_sensitivity(train: pd.DataFrame, seed: int, **fit_kw: int) -> dict[str, Any]:
    """Research plan validity checks: traffic threshold and fuel-effect prior."""
    from pitwall.models import tire_degradation as td

    def summary(df: pd.DataFrame, **kw: Any) -> dict[str, Any]:
        idata, _ = td.fit(df, seed=seed, **fit_kw, **kw)
        post = td.posterior_draws(idata)
        return {
            "beta_compound": dict(
                zip(post["compounds"], map(float, post["beta_compound"].mean(axis=0)), strict=True)
            ),
            "fuel_effect": float(post["fuel_effect"].mean()),
            "diagnostics": td.diagnostics(idata),
        }

    out = {}
    for gap in (1.0, 2.5):
        df = train.assign(in_traffic=train["gap_to_car_ahead_s"].fillna(np.inf) < gap)
        out[f"traffic_gap_{gap}s"] = summary(df)
    out["fuel_prior_wide"] = summary(train, fuel_prior=(0.04, 0.06))
    return out


# --- RQ2 -------------------------------------------------------------------------------------


def circuit_deg_tiers() -> pd.DataFrame:
    """Circuit degradation tier (low/mid/high tertile) from the RQ1 posterior mean slope of
    the race compounds (MEDIUM, HARD), as a circuit attribute for RQ2 heterogeneity."""
    post = _read("features", "rq1_degradation_posterior").toPandas()
    race = post[post["compound"].isin(["MEDIUM", "HARD"])]
    mean = race.groupby("circuit_id")["beta_s_per_lap"].mean()
    tiers = pd.qcut(mean, 3, labels=["low", "mid", "high"]).astype(str)
    return tiers.rename("deg_tier").reset_index()


RQ2_TABLES = {"cycle": "feat_pit_cycle", "lap": "feat_pit_decision"}


def build_feat_rq2(season: int) -> dict[str, int]:
    """Both RQ2 designs: feat_pit_cycle (primary) and feat_pit_decision (lap-level)."""
    from pitwall.features.pit_decision import (
        build_feat_pit_cycle,
        build_feat_pit_decision,
        circuit_pit_loss,
    )

    spark = get_spark()
    in_season = F.col("season") == season
    sessions = _read("gold", "dim_session")
    try:
        tiers = spark.createDataFrame(circuit_deg_tiers())
    except Exception:  # noqa: BLE001 - RQ1 not run yet: no heterogeneity tier
        log.warning("No RQ1 posterior found; deg_tier will be NULL")
        tiers = None
    laps = _read("gold", "fact_lap").filter(in_season)
    loss = circuit_pit_loss(_read("gold", "fact_pit_stop"), sessions)
    counts = {}
    for design, build in [("cycle", build_feat_pit_cycle), ("lap", build_feat_pit_decision)]:
        feat = build(laps, sessions.filter(in_season), loss, tiers)
        path = table_path("features", RQ2_TABLES[design])
        overwrite_season(spark, feat, path, season)
        counts[design] = _read("features", RQ2_TABLES[design]).filter(in_season).count()
    return counts


def _propensity_plot(t: np.ndarray, e_hat: np.ndarray, path: Path, labels: tuple) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(e_hat[t == 0], bins=bins, alpha=0.6, density=True, label=labels[0])
    ax.hist(e_hat[t == 1], bins=bins, alpha=0.6, density=True, label=labels[1])
    for x in (0.05, 0.95):
        ax.axvline(x, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Estimated propensity (cross-fitted by race)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _rq2_analysis(df: pd.DataFrame, seed: int, n_boot: int, n_perm: int) -> dict[str, Any]:
    """All estimators, diagnostics and refutations for one design's units."""
    from pitwall.models import causal

    primary, secondary, placebo = (
        "outcome_ahead_after_cycle",
        "outcome_gap_gain_s",
        "placebo_gap_gain_before_s",
    )
    res: dict[str, Any] = {
        "n_units": len(df),
        "n_treated": int(df["treated"].sum()),
        "n_races": int(df["race_id"].nunique()),
        "p_ahead_treated_raw": float(df.loc[df["treated"] == 1, primary].mean()),
        "p_ahead_control_raw": float(df.loc[df["treated"] == 0, primary].mean()),
    }
    e_hat, g_hat = causal.cross_fit(df, primary, seed=seed)
    keep = causal.overlap_mask(e_hat)
    res["overlap"] = {
        "share_kept": float(keep.mean()),
        "treated_kept": int(df["treated"].to_numpy()[keep].sum()),
        "control_kept": int((1 - df["treated"].to_numpy())[keep].sum()),
    }
    res["primary"] = {
        "naive": causal.naive(df, primary).as_dict(),
        "ipw": causal.ipw(df, primary, e_hat, n_boot=n_boot, seed=seed).as_dict(),
        "dml": causal.dml(df, primary, e_hat, g_hat).as_dict(),
        "econml_linear_dml": causal.econml_cross_check(df[keep], primary, seed),
    }
    rr = causal.risk_ratio(df, primary, e_hat)
    p0 = float(df.loc[keep & (df["treated"] == 0).to_numpy(), primary].mean())
    dml_est = res["primary"]["dml"]
    # E-value for the CI bound closest to the null (VanderWeele & Ding).
    bound = dml_est["lo"] if dml_est["effect"] > 0 else dml_est["hi"]
    crosses_null = dml_est["lo"] <= 0 <= dml_est["hi"]
    res["sensitivity"] = {
        "risk_ratio": rr,
        "e_value": causal.e_value(rr),
        "e_value_ci": 1.0 if crosses_null else causal.e_value((p0 + bound) / p0),
    }

    sec = df.dropna(subset=[secondary])
    e2, g2 = causal.cross_fit(sec, secondary, seed=seed)
    res["secondary_gap_gain_s"] = causal.dml(sec, secondary, e2, g2).as_dict()
    pl = df.dropna(subset=[placebo])
    e3, g3 = causal.cross_fit(pl, placebo, seed=seed)
    res["placebo_gap_gain_before_s"] = causal.dml(pl, placebo, e3, g3).as_dict()
    res["permuted_treatment"] = causal.permuted_treatment_refuter(df, primary, n_perm, seed)

    weights = causal.ipw_weights(df["treated"].to_numpy(), e_hat)
    smd_before = causal.smd(df[keep])
    smd_after = causal.smd(df[keep], weights[keep])
    res["balance"] = {
        "max_abs_smd_before": float(smd_before.abs().max()),
        "max_abs_smd_after": float(smd_after.abs().max()),
        "share_covariates_balanced_after": float((smd_after.abs() < 0.1).mean()),
        "smd_before": smd_before.round(3).to_dict(),
        "smd_after": smd_after.round(3).to_dict(),
    }
    res["by_deg_tier"] = {}
    for tier in sorted(df["deg_tier"].dropna().unique()):
        mask = (df["deg_tier"] == tier).to_numpy()
        n_t = int(df.loc[mask, "treated"].sum())
        if n_t >= 20 and mask.sum() - n_t >= 20:
            res["by_deg_tier"][tier] = causal.dml(
                df[mask], primary, e_hat[mask], g_hat[mask]
            ).as_dict()
    res["_e_hat"] = e_hat
    return res


def run_rq2(seasons: list[int], seed: int = 42, n_boot: int = 500, n_perm: int = 20) -> dict:
    """Research plan §RQ2 end to end, both designs. Writes reports/rq2/* and an MLflow run."""
    import mlflow

    from pitwall.tracking import tracked_run

    out = REPORTS / "rq2"
    frames = {
        design: _read("features", table)
        .filter(F.col("season").isin(seasons))
        .toPandas()
        .dropna(subset=["outcome_ahead_after_cycle"])
        .reset_index(drop=True)
        for design, table in RQ2_TABLES.items()
    }
    params = {
        "seasons": ",".join(map(str, seasons)),
        **{f"n_units_{d}": len(f) for d, f in frames.items()},
    }
    tables = {t: table_path("features", t) for t in RQ2_TABLES.values()}
    with tracked_run("rq2-undercut", params, tables, seed) as run:
        res: dict[str, Any] = {"mlflow_run_id": run.info.run_id, "seasons": seasons}
        for design, df in frames.items():
            analysis = _rq2_analysis(df, seed, n_boot, n_perm)
            e_hat = analysis.pop("_e_hat")
            labels = (
                ("rival stopped first", "stopped first (undercut)")
                if design == "cycle"
                else ("did not pit that lap", "pitted that lap")
            )
            _propensity_plot(
                df["treated"].to_numpy(), e_hat, out / f"propensity_overlap_{design}.png", labels
            )
            res[design] = analysis
            for method in ("naive", "ipw", "dml"):
                est = analysis["primary"][method]
                mlflow.log_metrics(
                    {f"{design}_{method}_{k}": est[k] for k in ("effect", "lo", "hi")}
                )
            mlflow.log_metrics({f"{design}_{k}": v for k, v in analysis["sensitivity"].items()})
        mlflow.log_artifacts(str(out), "rq2")
        _write_json(out / "results.json", res)
    return res


# --- RQ3 -------------------------------------------------------------------------------------


def pit_loss_per_stop() -> pd.DataFrame:
    """Empirical pit loss per green-flag stop: in-lap + out-lap - 2 x the driver's median
    clean lap in that race. Spark in, pandas out (one row per stop)."""
    from pyspark.sql import Window

    from pitwall.gold.lap_facts import clean_lap_flag

    laps = _read("gold", "fact_lap").filter(F.col("session_type") == "R")
    key = ["season", "event_round", "driver_id"]
    median = (
        laps.filter(clean_lap_flag())
        .groupBy(*key)
        .agg(F.percentile_approx("lap_time_s", 0.5).alias("_median_s"))
    )
    nxt = Window.partitionBy(*key).orderBy("lap_number")
    races = (
        _read("gold", "dim_session")
        .filter(F.col("session_type") == "R")
        .select("season", F.col("round").alias("event_round"), "circuit_id")
    )
    stops = (
        laps.withColumn("_out_time", F.lead("lap_time_s").over(nxt))
        .withColumn("_out_neutral", F.lead("is_neutralized").over(nxt))
        .filter(F.col("is_pit_in_lap") & ~F.col("is_neutralized") & ~F.col("_out_neutral"))
        .join(median, key)
        .join(races, ["season", "event_round"])
        .select(
            "season",
            "circuit_id",
            (F.col("lap_time_s") + F.col("_out_time") - 2 * F.col("_median_s")).alias("pit_loss_s"),
        )
        .filter(F.col("pit_loss_s").isNotNull())
    )
    return stops.toPandas()


def sc_events_per_race() -> pd.DataFrame:
    """SC and VSC deployments per race with lap-level data, and the race distance."""
    control = _read("silver", "race_control").filter(F.col("session_type") == "R")
    covered = control.select("season", F.col("event_round").alias("round")).distinct()
    counts = (
        control.filter(F.col("status").isin("4", "6"))
        .groupBy("season", F.col("event_round").alias("round"))
        .agg(F.count(F.lit(1)).alias("n_sc_starts"))
    )
    races = _read("gold", "dim_session").filter(F.col("session_type") == "R")
    return (
        races.join(covered, ["season", "round"])
        .join(counts, ["season", "round"], "left")
        .select(
            "season",
            "circuit_id",
            F.col("scheduled_laps").alias("laps"),
            F.coalesce("n_sc_starts", F.lit(0)).alias("n_sc_starts"),
        )
        .toPandas()
    )


def load_race_inputs(season: int) -> list:
    from pitwall.features.race_sim_params import build_race_inputs

    sessions = _read("gold", "dim_session").filter(F.col("session_type") == "R").toPandas()
    results = _read("gold", "fact_session_result").filter(F.col("season") == season).toPandas()
    stints = (
        _read("gold", "fact_stint")
        .filter((F.col("season") == season) & (F.col("session_type") == "R"))
        .toPandas()
    )
    posterior = _read("features", "rq1_degradation_posterior").toPandas()
    return build_race_inputs(
        season, sessions, results, stints, pit_loss_per_stop(), sc_events_per_race(), posterior
    )


def save_race_inputs(inputs: list, season: int) -> dict[str, int]:
    """features.feat_race_sim_params (race grain) + feat_race_sim_cars (car grain)."""
    from pitwall.features.race_sim_params import to_tables

    spark = get_spark()
    races, cars = to_tables(inputs, season)
    for name, pdf in [("feat_race_sim_params", races), ("feat_race_sim_cars", cars)]:
        overwrite_season(spark, spark.createDataFrame(pdf), table_path("features", name), season)
    return {"races": len(races), "cars": len(cars)}


def build_feat_rq3(season: int) -> dict[str, int]:
    return save_race_inputs(load_race_inputs(season), season)


def _plot_rq3(
    reliability: pd.DataFrame, pit: np.ndarray, scaling: pd.DataFrame, gains: pd.Series, out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot([0, 1], [0, 1], color="black", lw=0.7, ls="--")
    axes[0].plot(reliability["mean_p"], reliability["observed"], "o-", color="#4c72b0")
    axes[0].set_xlabel("Predicted P(top 10)")
    axes[0].set_ylabel("Observed frequency")
    axes[0].set_title("Reliability, backtest")
    axes[1].hist(pit, bins=10, color="#55a868", edgecolor="white")
    axes[1].axhline(len(pit) / 10, color="black", lw=0.7, ls="--")
    axes[1].set_title("PIT of actual finishing position")
    fig.tight_layout()
    fig.savefig(out / "backtest_calibration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(scaling["parallelism"], scaling["speedup"], "o-", label="measured")
    ax.plot(
        scaling["parallelism"],
        scaling["parallelism"] / scaling["parallelism"].iloc[0],
        ls="--",
        color="grey",
        label="linear",
    )
    ax.set_xlabel("Concurrent Spark tasks (cores)")
    ax.set_ylabel("Speed-up")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.legend(frameon=False)
    ax.set_title("Simulation scaling")
    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(gains, bins=30, color="#c44e52", edgecolor="white")
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Expected points gained by the best alternative (held-out sims)")
    ax.set_ylabel("driver-races")
    fig.tight_layout()
    fig.savefig(out / "decision_gains.png", dpi=150)
    plt.close(fig)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n: int = 2000) -> list[float]:
    means = [rng.choice(values, len(values)).mean() for _ in range(n)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run_rq3(
    season: int = 2025,
    backtest_sims: int = 10_000,
    decision_sims: int = 800,
    benchmark_sims: int = 200_000,
    million_run: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Research plan §RQ3: backtest, decision analysis, scaling benchmark."""
    import time

    import mlflow

    from pitwall.simulation import distributed as dist
    from pitwall.simulation import scoring
    from pitwall.simulation.strategies import candidate_strategies
    from pitwall.tracking import tracked_run

    out = REPORTS / "rq3"
    out.mkdir(parents=True, exist_ok=True)
    spark = get_spark()
    inputs = load_race_inputs(season)
    save_race_inputs(inputs, season)
    races = [ri.setup for ri in inputs]
    params = {
        "season": season,
        "n_races": len(races),
        "backtest_sims": backtest_sims,
        "decision_sims": decision_sims,
    }
    tables = {
        t: table_path(layer, t)  # type: ignore[arg-type]
        for layer, t in [
            ("gold", "fact_lap"),
            ("gold", "fact_stint"),
            ("gold", "fact_session_result"),
            ("features", "rq1_degradation_posterior"),
        ]
    }
    rng = np.random.default_rng(seed)
    with tracked_run("rq3-strategy", params, tables, seed) as run:
        res: dict[str, Any] = {"mlflow_run_id": run.info.run_id, **params}

        # Backtest: simulate every race with the strategies actually run, score vs grid.
        counts = dist.simulate_on_spark(spark, races, backtest_sims, seed=seed)
        per_race, pits, p10, hit10, pwin, won, grid1 = [], [], [], [], [], [], []
        rps_model, rps_grid = [], []
        for ri in inputs:
            n = len(ri.driver_ids)
            probs = dist.position_probs(counts, ri.setup.race_id, n)
            grid_rank = np.argsort(np.argsort(np.where(ri.grid > 0, ri.grid, 99))) + 1
            r_m = scoring.rps(probs, ri.actual_finish)
            r_g = scoring.rps(scoring.degenerate(grid_rank, n), ri.actual_finish)
            rps_model.append(r_m)
            rps_grid.append(r_g)
            pits.append(scoring.randomized_pit(probs, ri.actual_finish, rng))
            p10.append(probs[:, :10].sum(axis=1))
            hit10.append(ri.actual_finish <= 10)
            pwin.append(probs[:, 0])
            won.append(ri.actual_finish == 1)
            grid1.append(grid_rank == 1)
            per_race.append(
                {
                    "race_id": ri.setup.race_id,
                    "circuit_id": ri.circuit_id,
                    "rps_model": float(r_m.mean()),
                    "rps_grid": float(r_g.mean()),
                }
            )
        race_gain = np.array([r["rps_grid"] - r["rps_model"] for r in per_race])
        rel = scoring.reliability(np.concatenate(p10), np.concatenate(hit10))
        p_win, y_win = np.concatenate(pwin), np.concatenate(won).astype(float)
        res["backtest"] = {
            "rps_model": float(np.concatenate(rps_model).mean()),
            "rps_grid_baseline": float(np.concatenate(rps_grid).mean()),
            "rps_improvement_per_race_mean": float(race_gain.mean()),
            "rps_improvement_ci": _bootstrap_ci(race_gain, rng),
            "share_races_model_better": float((race_gain > 0).mean()),
            "top10_ece": scoring.expected_calibration_error(rel),
            "win_brier": float(np.mean((p_win - y_win) ** 2)),
            "win_brier_grid": float(np.mean((np.concatenate(grid1) - y_win) ** 2)),
            "per_race": per_race,
        }
        rel.to_csv(out / "top10_reliability.csv", index=False)

        # Decision analysis: best alternative per driver, chosen on split 0, scored on split 1.
        candidates = {
            r.race_id: candidate_strategies(r.params.race_laps, min_stint=10, step=3) for r in races
        }
        targets = {r.race_id: list(range(len(r.cars))) for r in races}
        sums = dist.evaluate_strategies_on_spark(
            spark,
            races,
            targets,
            candidates,
            decision_sims,
            batch_size=decision_sims // 2,
            seed=seed + 1,
        )
        choose, score = dist.summarize(sums, split=0), dist.summarize(sums, split=1)
        best = (
            choose[~choose["is_actual"]]
            .sort_values("expected_points", ascending=False)
            .groupby(["race_id", "car_index"])
            .head(1)[["race_id", "car_index", "strategy"]]
        )
        held = best.merge(score, on=["race_id", "car_index", "strategy"])
        actual_pts = score[score["is_actual"]][["race_id", "car_index", "expected_points"]]
        held = held.merge(
            actual_pts.rename(columns={"expected_points": "actual_points"}),
            on=["race_id", "car_index"],
        )
        names = {(ri.setup.race_id, i): d for ri in inputs for i, d in enumerate(ri.driver_ids)}
        held["driver_id"] = [
            names[(r, c)] for r, c in zip(held["race_id"], held["car_index"], strict=True)
        ]
        held.to_csv(out / "decision_analysis.csv", index=False, float_format="%.4f")
        gain = held["points_gain_vs_actual"]
        by_race = held.groupby("race_id")["points_gain_vs_actual"].mean().to_numpy()
        res["decision"] = {
            "n_candidates_per_race_mean": float(np.mean([len(c) for c in candidates.values()])),
            "n_driver_races": len(held),
            "mean_gain_per_driver_race": float(gain.mean()),
            "mean_gain_ci": _bootstrap_ci(by_race, rng),
            "share_gain_below_0_5_pts": float((gain < 0.5).mean()),
            "share_gain_over_2_se": float((gain > 2 * held["points_gain_se"]).mean()),
            "top_gains": held.nlargest(10, "points_gain_vs_actual")[
                [
                    "race_id",
                    "driver_id",
                    "strategy",
                    "actual_points",
                    "expected_points",
                    "points_gain_vs_actual",
                    "points_gain_se",
                    "p_better_than_actual",
                ]
            ]
            .round(3)
            .to_dict("records"),
        }

        # Scaling: same work at increasing parallelism, then one 10^6-race run.
        cores = spark.sparkContext.defaultParallelism
        levels = [p for p in (1, 2, 4, 8, 16, 32) if p <= cores]
        scaling = dist.scaling_benchmark(spark, races[0], benchmark_sims, levels)
        scaling.to_csv(out / "scaling.csv", index=False)
        res["scaling"] = scaling.to_dict("records")
        if million_run:
            start = time.perf_counter()
            big = dist.simulate_on_spark(spark, [races[0]], 1_000_000, batch_size=5000, seed=seed)
            elapsed = time.perf_counter() - start
            probs = dist.position_probs(big, races[0].race_id, len(races[0].cars))
            top = np.argsort(-probs[:, 0])[:3]
            res["million_run"] = {
                "race_id": races[0].race_id,
                "n_sims": 1_000_000,
                "seconds": elapsed,
                "cores": cores,
                "p_win_top3": {inputs[0].driver_ids[i]: float(probs[i, 0]) for i in top},
            }
        _plot_rq3(rel, np.concatenate(pits), scaling, gain, out)

        mlflow.log_metrics({k: v for k, v in res["backtest"].items() if isinstance(v, float)})
        mlflow.log_metrics({k: v for k, v in res["decision"].items() if isinstance(v, float)})
        mlflow.log_artifacts(str(out), "rq3")
        _write_json(out / "results.json", res)
    return res


# --- EDA (research plan §3) ------------------------------------------------------------------


def run_eda(seasons: list[int]) -> dict[str, Any]:
    """Coverage audit, lap-time and stint distributions, pit-loss sanity check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pitwall.gold.lap_facts import clean_lap_flag

    out = REPORTS / "eda"
    out.mkdir(parents=True, exist_ok=True)
    in_seasons = F.col("season").isin(seasons)
    laps = _read("gold", "fact_lap").filter(in_seasons & (F.col("session_type") == "R"))
    races = _read("gold", "dim_session").filter(F.col("session_type") == "R")

    coverage = (
        laps.groupBy("season")
        .agg(
            F.countDistinct("event_round").alias("races"),
            F.count(F.lit(1)).alias("laps"),
            F.avg(F.col("lap_time_s").isNull().cast("int")).alias("share_no_time"),
            F.avg(F.col("is_accurate").cast("int")).alias("share_accurate"),
            F.avg(F.col("is_green_flag").cast("int")).alias("share_green"),
            F.avg((F.col("compound") == "UNKNOWN").cast("int")).alias("share_unknown_compound"),
            F.avg(F.col("track_temp_c").isNull().cast("int")).alias("share_no_weather"),
            F.avg(clean_lap_flag().cast("int")).alias("share_clean"),
        )
        .orderBy("season")
        .toPandas()
    )
    scheduled = (
        races.filter(in_seasons).groupBy("season").agg(F.count(F.lit(1)).alias("races_scheduled"))
    ).toPandas()
    coverage = coverage.merge(scheduled, on="season", how="left")
    coverage.to_csv(out / "coverage.csv", index=False, float_format="%.4f")

    clean = laps.filter(clean_lap_flag() & F.col("compound").isin("SOFT", "MEDIUM", "HARD"))
    sample = (
        clean.join(
            races.select("season", F.col("round").alias("event_round"), "circuit_id"),
            ["season", "event_round"],
        )
        .select("season", "circuit_id", "compound", "lap_time_s", "tyre_life_laps")
        .toPandas()
    )
    sample["rel"] = sample["lap_time_s"] / sample.groupby(["season", "circuit_id"])[
        "lap_time_s"
    ].transform("median")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for compound, color in [("SOFT", "#d62728"), ("MEDIUM", "#e6b800"), ("HARD", "#7f7f7f")]:
        part = sample[sample["compound"] == compound]
        axes[0].hist(
            part["rel"].clip(0.97, 1.06),
            bins=80,
            histtype="step",
            color=color,
            label=compound,
            density=True,
        )
        age = part.groupby("tyre_life_laps")["rel"].median()
        age = age[part.groupby("tyre_life_laps").size() >= 200]
        axes[1].plot(age.index, age.values, color=color, label=compound)
    axes[0].set_xlabel("Lap time / race median (clean laps)")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Tyre age (laps)")
    axes[1].set_ylabel("Median relative lap time")
    axes[1].set_title("Raw (fuel burn hides wear: later laps are lighter)")
    fig.tight_layout()
    fig.savefig(out / "lap_times_by_compound.png", dpi=150)
    plt.close(fig)

    stints = _read("gold", "fact_stint").filter(in_seasons & (F.col("session_type") == "R"))
    strategies = (
        stints.groupBy("season", "event_round", "driver_id")
        .agg(
            F.count(F.lit(1)).alias("stints"),
            F.sort_array(F.collect_list(F.struct("stint_number", "compound"))).alias("s"),
        )
        .withColumn("stops", F.col("stints") - 1)
        .toPandas()
    )
    stops = strategies["stops"].clip(upper=4).value_counts(normalize=True).sort_index()
    stint_laps = stints.select("compound", "laps").toPandas()
    stint_len = (
        stint_laps[stint_laps["compound"].isin(["SOFT", "MEDIUM", "HARD"])]
        .groupby("compound")["laps"]
        .describe()[["count", "mean", "50%", "max"]]
    )

    pit = pit_loss_per_stop()
    pit = pit[pit["season"].isin(seasons) & pit["pit_loss_s"].between(5, 60)]
    pit_by_circuit = (
        pit.groupby("circuit_id")["pit_loss_s"].agg(["median", "count"]).sort_values("median")
    )
    pit_by_circuit.to_csv(out / "pit_loss_by_circuit.csv", float_format="%.2f")

    result = {
        "seasons": seasons,
        "coverage": coverage.to_dict("records"),
        "stops_share": {str(k): float(v) for k, v in stops.items()},
        "stint_laps_by_compound": stint_len.round(2).to_dict("index"),
        "pit_loss_median_s": float(pit["pit_loss_s"].median()),
        "pit_loss_iqr_s": [
            float(pit["pit_loss_s"].quantile(0.25)),
            float(pit["pit_loss_s"].quantile(0.75)),
        ],
        "pit_loss_lowest": pit_by_circuit.head(3)["median"].round(2).to_dict(),
        "pit_loss_highest": pit_by_circuit.tail(3)["median"].round(2).to_dict(),
    }
    _write_json(out / "results.json", result)
    return result
