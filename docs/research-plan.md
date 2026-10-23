# Research plan

This document is written **before** modeling, in the spirit of a pre-registration. For each question it fixes the estimand, the method, the baseline and the evaluation in advance. Then results can't quietly drift toward whatever looks best. Deviations get logged in §6.

## 1. Research questions

### RQ1 — Tire degradation

| | |
|---|---|
| **Question** | How much lap time does each compound lose per lap of tire age, and how does that vary by circuit and track temperature? |
| **Hypotheses** | H1a: degradation (s/lap) is ordered SOFT > MEDIUM > HARD at most circuits. H1b: degradation increases with track temperature. H1c: circuits vary more than drivers do. |
| **Estimand** | β_c,k = expected change in fuel-corrected lap time per extra lap of tire age, for compound *k* at circuit *c* (green-flag, clean-air laps). |
| **Data** | `feat_tire_degradation`, 2023–2024 train, 2025 test (lap data starts in 2023; see §6). |
| **Model** | Bayesian hierarchical regression (PyMC): `lap_time ~ race_offset + driver_offset + fuel_effect_c · fuel_proxy + (β_k + β_c,k) · tyre_life + γ · track_temp · tyre_life + δ · in_traffic`, with partial pooling of β_c,k across circuits and a Student-t likelihood (robust to outliers). Non-linear "cliff" handled by a stretch extension (piecewise/spline in tyre_life). |
| **Baselines** | (B0) per-stint OLS slope averaged by compound; (B1) LightGBM on the same features. |
| **Evaluation** | Predict the lap times of held-out 2025 stints given their first 3 laps: RMSE/MAE vs baselines; **90% interval coverage**; posterior predictive checks; R-hat < 1.01, ESS > 400, no divergences. |
| **Validity checks** | Parameter recovery on simulated data; sensitivity to the traffic threshold and fuel-effect prior. |
| **Deliverable** | Degradation table + forest plot by circuit and compound; posterior samples exported for RQ3. |

### RQ2 — Does the undercut work?

| | |
|---|---|
| **Question** | When two cars are fighting (gap < 3 s), does pitting first *cause* the car behind to end up ahead after both have pitted? |
| **Estimand** | ATE (and CATE by circuit degradation tier) of *pitting first at lap t* on P(ahead of rival after the pit cycle). |
| **Unit** | Driver–rival pair × lap within a pit window, from `feat_pit_decision`. |
| **Identification** | Selection on observables: pitting decisions depend on gap, tire age/compound, laps remaining, degradation, expected rejoin traffic, all of which are observed at lap *t*. Draw the assumed DAG in the report. |
| **Methods** | (1) Naive difference (to show the bias); (2) inverse propensity weighting with overlap trimming; (3) **double ML** (EconML `LinearDML` / `CausalForestDML`) with cross-fitting; check that the methods agree. |
| **Diagnostics** | Propensity overlap plots, covariate balance (SMD < 0.1), cluster-robust CIs by race. |
| **Robustness** | Placebo outcome (gap change *before* the stop); placebo treatment timing; E-value sensitivity to unobserved confounding; exclude SC-disrupted cycles. |
| **Threats** | Team orders, damage, unobserved tire state, strategic interaction (the rival reacts). The report states these explicitly. |
| **Deliverable** | Effect estimate with CI, heterogeneity plot by degradation tier, plain-language conclusion. |

### RQ3 — Optimal strategy under uncertainty

| | |
|---|---|
| **Question** | For a given race, what pit strategy maximizes expected points, how uncertain is that, and how often did teams pick it? |
| **Simulator** | Lap-by-lap race model: base pace + degradation (RQ1 posterior draws) + fuel effect + pit loss (empirical per circuit) + SC/VSC events (hazard per lap from history) + simple overtaking model (pace delta and DRS threshold) + undercut effect (RQ2). |
| **Compute** | Spark: broadcast parameters; distribute `(race, strategy, batch_of_sims)` over executors; target 10⁶+ simulated races per study. Scaling benchmark: 1 → N workers. |
| **Validation (backtest)** | Simulate each 2025 race using the strategies teams actually ran. Score the predicted distribution of finishing positions against actual results: **calibration**, CRPS/rank-probability score vs a "finish = grid" baseline. |
| **Decision analysis** | For each driver/race, compare the actual strategy against the simulator's best: expected points lost, P(better). |
| **Deliverable** | Strategy explorer app + "teams that deviated lost X points (CI)" finding. |

### RQ4 (stretch) — Calibrated pre-race predictions

| | |
|---|---|
| **Target** | P(podium), P(points) per driver, before the race starts. |
| **Model** | Gradient boosting + isotonic/Platt calibration; compare with ordinal/rank model. |
| **Baseline** | Grid-position-only logistic model. |
| **Metrics** | Brier score, log loss, reliability diagrams, expected calibration error. Rolling-origin evaluation by race. |

## 2. Cross-cutting methodology

- **Temporal validation only.** Train ≤ season Y−1, test on season Y; rolling-origin backtests within a season where needed. Never random splits.
- **Leakage controls.** Point-in-time feature builders, tested. No post-decision information in RQ2 covariates.
- **Regulation eras.** 2022 brought major car regulation changes. All lap-level data (2023–2025) falls inside the 2022+ era, so era effects can't be estimated and no cross-era claims are made (see §6).
- **Uncertainty everywhere.** Posterior intervals, cluster-robust CIs, simulation distributions. No bare point estimates in the reports.
- **Baselines first.** Each RQ reports its baseline before the main model.
- **Tracking.** Every fit is an MLflow run with git SHA, Delta table versions, seed and config.

## 3. Exploratory analysis (before modeling)

1. Data coverage audit: missing laps/telemetry by season and session; FastF1 `IsAccurate` rates.
2. Lap-time distributions by compound and circuit; outlier taxonomy (SC, traffic, mistakes).
3. Stint-length and strategy distributions per circuit and era.
4. Pit-loss estimates per circuit vs published numbers (sanity check).

## 4. Reporting

Each RQ gets `reports/rqN_<name>.md` (rendered from a notebook) with: question → data → method → results (with uncertainty) → robustness → limitations → plain-language takeaway.

## 5. Definition of done (science)

- [ ] Baselines and main model evaluated on held-out season
- [ ] Diagnostics pass (convergence, balance, calibration as applicable)
- [ ] Robustness checks reported, including ones that weaken the result
- [ ] Limitations section written
- [ ] Result reproducible from `make` target + MLflow run ID

## 6. Deviations log

| Date | RQ | Change | Reason |
|---|---|---|---|
| 2026-10-22 | RQ1–RQ3 | Lap-level data window narrowed from 2018–2025 to **2023–2025**. RQ1 trains on 2023–2024 and tests on 2025; RQ2 and RQ3 use the same window. Jolpica results stay 2018+ for dimensions and history. The planned within-era check is dropped as moot. | The FastF1 timing archive needs an F1 TV login since Oct 2026. OpenF1 is free but starts in 2023 ([ADR-0006](adr/0006-openf1-as-lap-data-source.md)). Decided before any modeling. |
