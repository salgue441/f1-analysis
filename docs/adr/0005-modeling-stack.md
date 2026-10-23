# ADR-0005: Modeling stack — PyMC, EconML/DoWhy, MLflow

- **Status:** Accepted
- **Date:** 2026-10-22

## Context
The research questions need (1) hierarchical models with honest uncertainty, (2) causal effect estimation with diagnostics, (3) experiment tracking for reproducibility.

## Decision
- **PyMC + ArviZ** for RQ1. Partial pooling handles circuits with few races; the posterior feeds directly into the RQ3 simulator.
- **EconML** (DML, causal forests) for estimation and **DoWhy** for DAG encoding + refutation tests in RQ2.
- **scikit-learn / LightGBM** for baselines and nuisance models.
- **MLflow** tracking for all fits (params, metrics, diagnostics plots, data versions, git SHA).

## Alternatives considered
- **Stan (CmdStanPy):** excellent, but an extra toolchain; PyMC keeps everything in Python.
- **statsmodels mixed models:** a good frequentist baseline (may be used as one), but no full posterior for simulation.
- **Weights & Biases:** hosted; MLflow runs locally and is common in Spark/Databricks shops.

## Consequences
- Heavy optional dependencies: grouped as extras (`.[ds]`, `.[causal]`) to keep CI fast.
