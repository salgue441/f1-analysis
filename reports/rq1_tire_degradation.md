# RQ1 — How fast do tyres wear?

**Takeaway.** Once fuel burn is accounted for, a lap of tyre age costs on average **0.065 s on the soft, 0.058 s on the medium and 0.040 s on the hard** (90% intervals below). The circuit matters far more than the compound or the driver. Mediums at Spa, Bahrain and the Red Bull Ring lose about 0.1 s per lap of age, while Baku's and Losail's lose almost nothing, and hotter tracks wear tyres faster. The hierarchical model gives the most accurate forecasts of how a 2025 stint will evolve from its first three laps. Its intervals are too narrow, though: 82% coverage where 90% was promised.

Run: MLflow `rq1-tire-degradation`, run `805542bc71fb4de7bcdd0adcd74bb400` · reproduce with `make rq1` · numbers from [`rq1/results.json`](rq1/results.json)

## Question and estimand

How much lap time does each compound lose per extra lap of tyre age, and how does that vary by circuit and track temperature? The estimand is β<sub>c,k</sub>: the expected change in fuel-corrected lap time per lap of tyre age, for compound *k* at circuit *c*, on green-flag dry race laps (research plan §RQ1).

## Data

`features.feat_tire_degradation`: race laps on dry compounds that are timed, green-flag, not in/out laps, not deleted and not lap 1. Laps slower than 115% of the weekend's pole time are also dropped; pole time is known before the race.

| | Seasons | Laps |
|---|---|---|
| Train | 2023–2024 | 41,033 |
| Test | 2025 | 21,012 (17,705 scored: all but each stint's first three laps; stints shorter than 8 laps skipped) |

Lap data covers 2023 onward only (OpenF1; [ADR-0006](../docs/adr/0006-openf1-as-lap-data-source.md)). Coverage is complete: every scheduled race is present and 87–89% of laps are clean ([EDA](eda/results.json)).

## Method

Lap time, centred on each race's median, is modeled as:

race offset + driver offset (sum-to-zero) + compound pace α<sub>c,k</sub> + fuel effect × laps remaining + (β<sub>c,k</sub> + driver wear deviation + γ × track temp) × tyre age + traffic effect + Student-t noise.

- β<sub>c,k</sub> = compound mean + circuit deviation, partially pooled across circuits (non-centred).
- α<sub>c,k</sub>, the fresh-tyre pace, is pooled the same way and sums to zero over compounds, since the overall level is the race offset's.
- Weakly informative priors (fuel 0.04 ± 0.02 s/lap). Fitted with NUTS (nutpie), 4 chains × 2000 draws after 2000 tuning steps.

Three changes from the pre-registered specification are logged in the research plan §6: the compound pace term, sum-to-zero driver offsets, and anchor noise in predictive intervals. All were made before these results were written up, for reasons unrelated to test performance.

**Evaluation (fixed before modeling).** Forecast each held-out 2025 stint from its first three laps. Every model uses the same rule: the anchor laps set the level (race, car, driver), and the model supplies the shape. We score RMSE, MAE and 90%-interval coverage.

**Baselines.**
- B0: median per-stint OLS slope by compound, with an empirical residual interval.
- B1: LightGBM on the same features, with quantile models for the interval.

## Results

### Forecasting held-out 2025 stints

| Model | RMSE (s) | MAE (s) | 90% coverage | Mean 90% width (s) |
|---|---|---|---|---|
| B0 per-stint slope | 1.226 | 0.892 | **89.7%** | 3.71 |
| B1 LightGBM | 1.285 | 0.945 | 65.7% | 2.19 |
| **Bayesian hierarchical** | **1.169** | **0.829** | 82.2% | 2.77 |

The Bayesian model has the best point forecasts: about 5% lower RMSE and 7% lower MAE than B0. LightGBM does worst out of sample. It learns circuit shapes from 2023–24 that don't carry over, and its quantile intervals are badly overconfident. The Bayesian intervals are 25% narrower than B0's, but they under-cover: 83% hard, 79% medium, 90% soft ([calibration plot](rq1/calibration.png)). **The model fails the 90% coverage target.** See Limitations.

### Degradation estimates

| Compound | Mean s/lap per lap of age | 90% interval |
|---|---|---|
| SOFT | 0.065 | 0.049 – 0.082 |
| MEDIUM | 0.058 | 0.045 – 0.071 |
| HARD | 0.040 | 0.027 – 0.053 |

![Degradation by circuit and compound](rq1/forest_plot.png)

Full table: [`rq1/degradation_table.csv`](rq1/degradation_table.csv). Other estimates (posterior mean, 90% interval):

- **Fresh-tyre pace** (relative to the compound average, race conditions): SOFT −0.01 (−0.11 to +0.09) s, MEDIUM −0.02 (−0.11 to +0.07) s, HARD +0.03 (−0.06 to +0.12) s. On average, compounds differ much less in race pace when new than in how fast they wear.
- **Fuel effect:** 0.0512 (0.0508–0.0516) s per lap of fuel still on board. A full 57-lap fuel load costs about 2.9 s per lap at the start, consistent with the 0.03–0.06 s range teams quote.
- **Traffic** (within 1.5 s of the car ahead): +0.30 (0.29–0.31) s per lap.
- **Circuit spread of degradation** (SD): 0.037 (0.031–0.044) s/lap per lap.
- **Driver spread of degradation** (SD): 0.0053 (0.0038–0.0073).
- **Temperature:** +0.0062 (0.0051–0.0073) s/lap per lap per SD of track temperature (8.0 °C). By the end of a 25-lap stint that's about 0.16 s more lost per lap for each 8 °C hotter.

### Hypotheses

| | Hypothesis | Posterior evidence | Verdict |
|---|---|---|---|
| H1a | SOFT > MEDIUM > HARD at most circuits | P(ordered) = 0.67 for the compound means, but > 0.5 at only 7 of 24 circuits | **Weak.** Soft and medium overlap. Pirelli picks a different C-number trio per circuit, so "soft" isn't the same tyre everywhere. |
| H1b | Degradation increases with track temperature | P(γ > 0) > 0.999 | **Supported** |
| H1c | Circuits vary more than drivers | P(circuit SD > driver SD) > 0.999; about 7× larger | **Supported** |

## Diagnostics and robustness

- **Convergence:** max R-hat 1.00, min bulk ESS 1,040, min tail ESS 1,959, no divergences. The research-plan bar is R-hat < 1.01 and ESS > 400.
- **Parameter recovery:** fitting simulated data with known parameters recovers the compound slopes, the soft-versus-hard pace gap, the fuel effect and ≥ 75% of circuit × compound slopes inside their 90% intervals (`tests/test_rq1.py`, slow suite).
- **Sensitivity:** refits with the traffic threshold at 1.0 s or 2.5 s, or a 3× wider fuel prior, move the compound slopes by at most 0.003 s/lap and the fuel effect by at most 0.0004.

  | Variant | SOFT | MEDIUM | HARD | Fuel |
  |---|---|---|---|---|
  | Main | 0.065 | 0.058 | 0.040 | 0.0512 |
  | Traffic gap 1.0 s | 0.066 | 0.057 | 0.040 | 0.0516 |
  | Traffic gap 2.5 s | 0.062 | 0.057 | 0.040 | 0.0509 |
  | Fuel prior SD 0.06 | 0.065 | 0.058 | 0.040 | 0.0512 |

- **Naive comparison:** raw per-stint slopes (B0) are 0.030 / 0.007 / 0.002 s/lap. Fuel burn masks most of the wear, which is why a fuel-corrected model is needed at all.

## Limitations

- **Intervals are too narrow (82% for a nominal 90%).** The model treats lap noise as independent, but a stint's errors are correlated: drivers manage pace and traffic comes and goes. A stint-level random slope or autocorrelated noise would likely fix this. We report the model as specified rather than tune it on 2025.
- **Track evolution is confounded with wear.** Rubber builds up during the race, so the track gets faster. With one global fuel effect, circuits where that's strong come out with near-zero or negative wear: Losail medium −0.013, Jeddah hard −0.004, both with intervals below zero. Read those as "wear smaller than track evolution plus fuel", not as tyres getting faster. The RQ3 simulator floors wear at zero.
- **Compound labels aren't comparable across circuits.** SOFT/MEDIUM/HARD map to different Pirelli compounds (C1–C6) per event. Per-circuit estimates are meaningful; cross-circuit compound comparisons less so.
- **Linear wear only.** The tyre "cliff" (non-linear drop-off) isn't modeled. That understates late-stint losses on long stints.
- **2023–2025 only, all in the 2022+ regulation era.** Nothing here generalizes across regulation changes (research plan §6).
