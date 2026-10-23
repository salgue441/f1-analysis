# ADR-0004: Dagster for orchestration

- **Status:** Accepted
- **Date:** 2026-10-22

## Context
Pipelines are naturally a graph of tables (bronze → silver → gold → features → models), partitioned by season, with frequent backfills.

## Decision
Use **Dagster software-defined assets**. Each table is an asset, partitioned by `season` (static partitions 2018–current). Run metadata (row counts, DQ pass rate, Delta version) is attached to each materialization.

## Alternatives considered
- **Airflow:** industry standard and worth knowing, but task-centric. Asset lineage and partition backfills take more work.
- **Databricks Workflows:** great on Databricks, ties the project to one vendor.
- **Makefile/cron only:** no lineage, retries or observability.

## Consequences
- Lineage graph and asset catalog come for free (good screenshots for the README).
- Business logic stays in plain `pitwall.*` functions; Dagster assets are thin wrappers, so the logic is testable without Dagster and portable to Airflow.
