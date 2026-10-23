# Roadmap

Phases are ordered so that **each one ends with something showable**. Phases 3–5 are the core of the project and get the most time.

| Phase | Goal | Key work | Definition of done | Portfolio artifact |
|---|---|---|---|---|
| **0. Foundation** ✅ | Repo, docs, dev environment | SAD, ADRs, research plan, Docker, CI, package skeleton | `make up && make test` passes | This repo structure |
| **1. Historical batch** | Jolpica → bronze → silver → gold dims | Rate-limited client, bronze writer, silver MERGE, `dim_*` incl. SCD2 `dim_driver_team`, `fact_session_result`, Dagster assets by season | Backfill 2018–2025 twice → identical counts; DQ report per run | ERD + Dagster lineage screenshot |
| **2. Laps & telemetry** | FastF1 → full star schema | `fact_lap`, `fact_stint`, `fact_pit_stop`, race control → flags, telemetry partitions, `driver_id_map`, OPTIMIZE/Z-order, skew analysis | All gold tables populated 2018–2025; perf write-up with Spark UI before/after | "Engineering notes" section in README |
| **3. EDA + RQ1** | Tire degradation model | EDA notebook, `feat_tire_degradation`, baselines, PyMC hierarchical model, PPCs, simulated-data recovery test | Science DoD met; `reports/rq1_tire_degradation.md` | Forest plot + findings |
| **4. RQ2** | Undercut causal effect | `feat_pit_decision`, DAG, naive vs IPW vs DML, balance, placebo & E-value | Science DoD met; `reports/rq2_undercut.md` | Causal write-up |
| **5. RQ3** | Strategy simulator at scale | Simulator core (NumPy), Spark distribution, backtest on 2025, decision analysis, scaling benchmark | Calibrated backtest; benchmark chart | "1M simulated races" finding |
| **6. Communicate** | Make it visible | Streamlit strategy explorer, README findings section, blog post | App runs from `make app`; post published | Blog + demo GIF |
| 7. Stretch | Extra signal | RQ4 calibrated predictions; OpenF1 live streaming (Structured Streaming); cloud run on EMR/Databricks | — | — |

## Suggested order of first tasks (Phase 1)

1. `make up`, `make test` green.
2. `pitwall ingest jolpica --season 2024 --endpoint results` → inspect bronze.
3. Implement `silver.results` + expectations; MERGE logic; unit tests.
4. `gold.dim_driver_team` (SCD2) from silver results (function already sketched).
5. Wrap steps 2–4 as Dagster assets partitioned by season; backfill.
