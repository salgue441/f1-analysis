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
