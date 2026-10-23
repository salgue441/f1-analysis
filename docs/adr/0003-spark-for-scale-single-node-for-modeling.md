# ADR-0003: Spark where data or compute is large; single-node for statistical modeling

- **Status:** Accepted
- **Date:** 2026-10-22

## Context
Telemetry (10⁸–10⁹ rows) and Monte Carlo simulation (10⁶+ races) need horizontal scale. Bayesian and causal modeling libraries (PyMC, EconML) are single-node, and the modeling datasets are small once aggregated (10⁵–10⁶ rows).

## Decision
- **PySpark** for ingestion, all layer transforms, telemetry feature extraction, per-group model fitting (`applyInPandas`) and the RQ3 simulation (distributed over `(race, strategy, batch)`).
- **pandas / PyMC / EconML** on aggregated feature tables (`toPandas()` after filtering), run on the driver or a single node.
- Prefer built-in Spark functions; then Pandas UDFs (Arrow); plain Python UDFs only with a written reason.

## Consequences
- Using Spark only where it pays off is itself the point, and the README explains it.
- Need a clear handoff boundary: feature tables are the contract between Spark and modeling code.
