# Roadmap

Phases are ordered so that **each one ends with something showable**. Phases 3–5 are the core of the project and get the most time.

Status: ✅ done · 🟡 done with a documented change of scope · ⬜ not started

| Phase | Goal | Key work | Definition of done | Status |
|---|---|---|---|---|
| **0. Foundation** | Repo, docs, dev environment | SAD, ADRs, research plan, Docker, CI, package skeleton | `make up && make test` passes | ✅ |
| **1. Historical batch** | Jolpica → bronze → silver → gold dims | Rate-limited client, bronze writer, silver MERGE, `dim_*` incl. SCD2 `dim_driver_team`, `fact_session_result`, Dagster assets by season | Backfill 2018–2025 twice → identical counts; DQ report per run | ✅ Backfilled 2018–2025. Rebuilding a season from bronze gives identical row counts **and checksums** for every silver/gold table. Every quality gate run is logged to `quarantine/_dq_runs`. [ERD](data-model.md#erd) and [lineage screenshot](img/dagster_lineage.png) (`make dagster`) |
| **2. Laps & telemetry** | Lap data → full star schema | `fact_lap`, `fact_stint`, `fact_pit_stop`, race control → flags, telemetry partitions, `driver_id_map`, OPTIMIZE/Z-order, skew analysis | All gold tables populated; perf write-up | 🟡 All tables populated from **OpenF1, 2023–2025**: the FastF1 archive needs an F1 TV login ([ADR-0006](adr/0006-openf1-as-lap-data-source.md)). Telemetry is loaded for the 2024 season (16.5M samples). The perf write-up uses measured layout evidence rather than Spark UI screenshots (README, Engineering notes) |
| **3. EDA + RQ1** | Tire degradation model | EDA notebook, `feat_tire_degradation`, baselines, PyMC hierarchical model, PPCs, simulated-data recovery test | Science DoD met; `reports/rq1_tire_degradation.md` | ✅ Convergence, recovery and robustness pass. Interval coverage misses the target (82% vs 90%), stated as a limitation |
| **4. RQ2** | Undercut causal effect | `feat_pit_cycle` / `feat_pit_decision`, DAG, naive vs IPW vs DML, balance, placebo & E-value | Science DoD met; `reports/rq2_undercut.md` | ✅ Primary design changed to pit cycles before the full run (research plan §6). The placebo is not null, and that is reported |
| **5. RQ3** | Strategy simulator at scale | Simulator core (NumPy), Spark distribution, backtest on 2025, decision analysis, scaling benchmark | Calibrated backtest; benchmark chart | 🟡 Simulator, Spark distribution, 2025 backtest (beats the grid baseline) and scaling benchmark are done, and 10⁶ races run in 20 s. **Calibration isn't met** (ECE 0.09, overconfident), and the decision analysis isn't credible; both are reported as such |
| **6. Communicate** | Make it visible | Streamlit strategy explorer, README findings section, blog post | App runs from `make app`; post published | ✅ App, README findings, demo GIF and [blog post](blog/f1-strategy-under-uncertainty.md). Cross-posting it to an external blog is up to the author |
| 7. Stretch | Extra signal | RQ4 calibrated predictions; OpenF1 live streaming (Structured Streaming); cloud run on EMR/Databricks | — | 🟡 **RQ4** done ([report](../reports/rq4_calibrated_predictions.md)): calibrated, but no better than the grid. That's a negative result. **Streaming** done in replay mode ([ADR-0007](adr/0007-race-day-streaming.md)); the live MQTT connector is written to OpenF1's documented interface but untested, since live data is a paid tier. **Cloud** is config-ready and verified on local S3 and inside the EMR image ([runbook](cloud.md), [ADR-0008](adr/0008-cloud-deployment.md)); no job has run in a real AWS or Databricks account |

## What would move the science forward next

1. **RQ1 intervals:** a stint-level random wear slope or AR(1) lap noise to fix under-coverage, plus a non-linear tyre "cliff" term.
2. **RQ1 track evolution:** a circuit-specific track-evolution or fuel term, to separate rubbering-in from tyre wear.
3. **RQ2:** a design that models the rival's response (sequential treatment), and team-orders exclusions.
4. **RQ3:** feed the RQ2 effect into the overtaking model as a calibration target, and tune the remaining simulator constants (overtaking threshold, SC duration) on the validation season.
5. **Data:** if FastF1 gains F1 TV authentication, re-enable the `fastf1_*` specs to extend lap-level data back to 2018.
6. **RQ4:** add practice long-run pace from OpenF1 (the information teams use to beat the grid), and a rank model with position-dependent noise so field-wide probabilities are coherent *and* calibrated.
7. **Streaming:** keep the race state in memory and write Delta less often to get below the ~8 s per-batch floor; run the live connector during a session.
