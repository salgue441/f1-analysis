# ADR-0001: Local-first Delta Lake lakehouse

- **Status:** Accepted
- **Date:** 2026-10-22

## Context
We need ACID tables, MERGE/upserts, schema evolution and time travel (for reproducible analyses) at $0 cost, with a path to cloud storage.

## Decision
Use **Delta Lake** (`delta-spark`) on the local filesystem (`./data/lakehouse`), mounted into Docker. The lakehouse root is a single config value (`lakehouse_root`), so moving to `s3a://` / `abfss://` needs no code changes.

## Alternatives considered
- **Apache Iceberg:** equally capable and more engine-neutral, but local setup with a catalog adds friction. Delta works with zero setup in PySpark.
- **Plain Parquet:** no MERGE, no ACID, no time travel.
- **Local object store (MinIO etc.):** more realistic, but extra moving parts and licensing/image churn. Can be added later behind the same config.
- **A warehouse (Snowflake/BigQuery):** cost; and it hides the distributed compute the project wants to show.

## Consequences
- Simple local dev; Delta time travel gives us data versions to log in MLflow.
- Spark must run in Docker/WSL2 on Windows (Hadoop native filesystem issues on Windows).
