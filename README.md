# 🏁 Pitwall — Quantifying F1 race strategy under uncertainty

> *How fast do tires really wear? Does the undercut actually work? What should teams have done?*
> Bayesian, causal and simulation models on a Spark + Delta Lake lakehouse built from 2018–2025 F1 timing and telemetry data.

## Key findings

*To be filled as each research question is completed. Each finding includes its uncertainty.*

| Question | Finding | Report |
|---|---|---|
| RQ1 — Tire degradation | — | `reports/rq1_tire_degradation.md` |
| RQ2 — Does the undercut work? | — | `reports/rq2_undercut.md` |
| RQ3 — Optimal strategy | — | `reports/rq3_strategy.md` |

## Approach

| | Method | Why |
|---|---|---|
| RQ1 | Hierarchical Bayesian regression (PyMC) | Circuits with few races borrow strength from others; full posterior feeds the simulator |
| RQ2 | Double ML + propensity weighting (EconML/DoWhy) | Pit timing is confounded; naive comparisons are biased |
| RQ3 | Monte Carlo simulation distributed on Spark | Decisions under uncertainty need distributions, not point estimates |

Full plan, with hypotheses and evaluation fixed before modeling: [docs/research-plan.md](docs/research-plan.md)

## Architecture

```
Jolpica API ─┐
FastF1 ──────┼─► Bronze (raw) ─► quality gates ─► Silver (clean) ─► Gold (star schema) ─► Features ─► Models ─► App / reports
OpenF1 ┄┄┄┄┄┘                       └─► Quarantine                                                  └─► MLflow
```

- **Lakehouse:** Delta Lake, medallion layers, MERGE-based idempotent loads, SCD2 dimensions
- **Compute:** PySpark (ETL, telemetry, simulation) + PyMC/EconML (modeling)
- **Orchestration:** Dagster assets partitioned by season
- **Docs:** [Architecture (SAD)](docs/SAD.md) · [Data model / ERD](docs/data-model.md) · [ADRs](docs/adr/) · [Roadmap](docs/roadmap.md)

## Quickstart

Requires Docker (on Windows, Docker Desktop with WSL2).

```bash
cp .env.example .env
make up              # build + start workbench and MLflow (:5000)
make test            # unit tests
make ingest-results SEASON=2024
docker compose exec workbench pitwall ingest fastf1 --season 2024 --round 1
docker compose exec workbench pitwall build silver-laps --season 2024
make dagster         # asset graph at http://localhost:3000
```

## Repository layout

```
conf/            configuration (base + per-environment)
docs/            SAD, data model, research plan, roadmap, ADRs
src/pitwall/     package: ingestion, bronze, silver, gold, features, quality, models, simulation, orchestration
tests/           unit tests (local Spark)
notebooks/       exploration (outputs stripped)
reports/         final write-ups per research question
app/             Streamlit strategy explorer
```

## Engineering notes

*Phase 2: partitioning, skew, file sizing and scaling benchmarks with Spark UI evidence.*

## Data & license

Data comes from [Jolpica-F1](https://github.com/jolpica/jolpica-f1), [FastF1](https://github.com/theOehrly/Fast-F1) and [OpenF1](https://openf1.org). It is unofficial and used for non-commercial, educational purposes. No raw data is committed to this repository. This project is not associated with Formula 1 or the FIA.
