# ADR-0008: Cloud deployment is configuration, verified on local S3

- **Status:** Accepted
- **Date:** 2026-10-23
- **Refines:** ADR-0001 (local-first lakehouse), SAD §8.2

## Context
SAD §2 sets a $0 budget and requires that moving to the cloud be a configuration change. Phase 7 asks for a cloud run on EMR or Databricks. A real run needs an account and spends money, so this phase delivers everything up to the run. It verifies what can be verified without one, so the remaining step is credentials and a command.

## Decision
- **Environment profiles, no code branches.** `conf/s3local.yaml` (S3-compatible store in Docker), `conf/aws.yaml` (S3 + EMR Serverless) and `conf/databricks.yaml` (Databricks job cluster) change only the lakehouse root and Spark settings. Two settings were added for managed platforms:
  - an empty `spark.master` leaves the platform's master in place;
  - `spark.delta_pip: false` uses the platform's Delta jars instead of downloading them.
  `spark.packages` adds `hadoop-aws` for `s3a://`.
- **PySpark is an optional extra (`spark`).** EMR and Databricks ship their own Spark builds, and a pip-installed PySpark would shadow them. The local image, CI and the Spark cluster install `.[spark,…]`.
- **AWS: EMR Serverless with a custom image.**
  - Terraform: a versioned, encrypted, private S3 bucket with lifecycle rules; an ECR repository the EMR service can pull from (scoped to the application); an EMR Serverless application on `emr-7.5.0` (Spark 3.5) with **capped capacity and a 15-minute auto-stop**; and a job role limited to the bucket.
  - The image is EMR's base plus Python 3.11 and the package.
  - `deploy/aws/run_job.sh` submits any `pitwall` command line.
- **Databricks: an Asset Bundle** (`databricks.yml`). It defines a three-task job (results → OpenF1 ingest → lap build) on runtime 15.4 LTS (Spark 3.5, Delta 3.2, Python 3.11, the same versions as local). It installs the package as a wheel without the `spark` extra, and its schedule is paused by default.
- **Verification without an account:**
  - `pitwall copy` moves 2024 bronze into an S3-compatible store (SeaweedFS in Docker).
  - The pipeline rebuilds silver and gold there through `s3a://`.
  - `pitwall fingerprints --season 2024` must match the local lakehouse table for table.
  - Terraform passes `terraform validate`, `run_job.sh` passes shellcheck, and `databricks.yml` validates against the Databricks CLI's bundle schema.
  - The EMR image builds from the public base, and pitwall runs inside it under EMR's own `spark-submit`, Spark and Delta. This check found that EMR's Delta jar must also go on `spark.submit.pyFiles` for the Python API (`delta.tables`); `run_job.sh` now passes it.
- **Ingestion stays where the network is.** Jolpica and OpenF1 are public HTTP APIs. If the cloud jobs have no internet egress (EMR Serverless without a VPC and NAT), ingest locally and `pitwall copy` bronze to S3. Silver and gold then build in the cloud from that bronze, the path verified above.

## Alternatives considered
- **EMR on EC2 or EKS:** more control, but a cluster to size and keep running. Serverless bills per job and stops itself, which suits weekly batch runs.
- **AWS Glue:** also serverless Spark, but it pins its own Python and library versions, and its job model differs from `spark-submit`. EMR Serverless runs the same entry point as local.
- **MinIO for the local S3 check:** the standard choice, but its community images are no longer published on Docker Hub or Quay. SeaweedFS's S3 gateway exercises the same `s3a://` client path.
- **A real cloud run now:** the only way to measure cloud performance, but it needs the author's account and spend. It is left as a one-command step in the runbook.

## Consequences
- Moving to S3 changes data paths only: `copy`, `fingerprints` and every pipeline step run unchanged against `s3a://`.
- Not covered until a real run:
  - IAM and networking specifics in the target account;
  - EMR Serverless itself (scheduling, S3 through EMRFS, pulling the image from ECR);
  - the Databricks job against a workspace.
  docs/cloud.md lists the commands.
- MLflow on the cloud needs `MLFLOW_TRACKING_URI` pointing at a server (e.g. SageMaker MLflow or Databricks' managed MLflow). The default local SQLite store doesn't apply there.
- Streaming (ADR-0007) stays local; its landing directory is a local path.
