# Running Pitwall in the cloud

The code is the same everywhere; only `conf/<env>.yaml` changes ([ADR-0008](adr/0008-cloud-deployment.md), SAD §8.2). This runbook goes from the local S3 check, which costs nothing, to AWS EMR Serverless and Databricks, which cost money while jobs run.

| Profile (`PITWALL_ENV`) | Storage | Spark | Delta |
|---|---|---|---|
| `local` (default) | `./data/lakehouse` | `local[*]` in Docker | pip `delta-spark` |
| `s3local` | `s3a://pitwall/lakehouse` on SeaweedFS in Docker | `local[*]` + `hadoop-aws` | pip `delta-spark` |
| `s3` | `s3a://<bucket>/lakehouse` on AWS S3 | `local[*]` + `hadoop-aws` | pip `delta-spark` |
| `aws` | `s3://<bucket>/lakehouse` | EMR Serverless (EMR 7.5, Spark 3.5) | EMR's built-in Delta |
| `databricks` | `s3://<bucket>/lakehouse` (external location) | job cluster, runtime 15.4 LTS | built into the runtime |

## 1. Local S3 (free): check the lakehouse works on object storage

```bash
docker compose --profile s3 up -d            # SeaweedFS S3 API on :8333, bucket "pitwall"
# copy one season's bronze from the local lakehouse (telemetry left out to keep it quick)
docker compose exec -e PITWALL_ENV=s3local workbench \
  pitwall copy --from /workspace/data/lakehouse --season 2024 \
  --exclude openf1_car_data --exclude openf1_location
# rebuild silver and gold from that bronze, on S3
docker compose exec -e PITWALL_ENV=s3local workbench pitwall run 2024 --no-ingest
docker compose exec -e PITWALL_ENV=s3local workbench pitwall run 2024 --no-ingest --laps
# compare with the local lakehouse: every season-2024 table should match
docker compose exec -e PITWALL_ENV=s3local workbench pitwall fingerprints --season 2024 > s3.json
docker compose exec workbench pitwall fingerprints --season 2024 > local.json
diff s3.json local.json && echo identical
```

The result of this check is recorded in [`reports/engineering/cloud_verification.json`](../reports/engineering/cloud_verification.json).

## 2. AWS: EMR Serverless

Before spending anything, the image can be checked locally. This runs pitwall with EMR's own Spark and Delta:

```bash
docker build -f deploy/aws/Dockerfile -t pitwall-emr:check .
make up && deploy/aws/smoke_test.sh 2024     # -> reports/engineering/emr_image_check.json
```

Prerequisites for the real thing: an AWS account, AWS CLI v2 with credentials, Terraform ≥ 1.6, Docker.

```bash
# Infrastructure: bucket, ECR repository, EMR Serverless application, job role
cd deploy/aws/terraform
terraform init
terraform apply -var bucket_name=<globally-unique-name>        # review the plan first
cd -

# Image: EMR's Spark + Python 3.11 + pitwall
repo=$(terraform -chdir=deploy/aws/terraform output -raw ecr_repository_url)
aws ecr get-login-password | docker login --username AWS --password-stdin "${repo%%/*}"
docker build -f deploy/aws/Dockerfile -t "$repo:latest" .
docker push "$repo:latest"

# Data: ingest locally (public APIs), then copy bronze up
root=$(terraform -chdir=deploy/aws/terraform output -raw lakehouse_root)
docker compose exec -e PITWALL_ENV=s3 -e PITWALL_LAKEHOUSE_ROOT="${root/s3:/s3a:}" \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN workbench \
  pitwall copy --from /workspace/data/lakehouse
# (s3a:// is Hadoop's S3 client; EMR reads the same objects as s3://)

# Jobs: any pitwall command line
deploy/aws/run_job.sh run 2025 --no-ingest --laps
deploy/aws/run_job.sh fingerprints --season 2025
aws emr-serverless get-job-run --application-id "$(terraform -chdir=deploy/aws/terraform output -raw application_id)" --job-run-id <id>
```

- **Cost controls:** the application can never use more than `max_vcpu` / `max_memory` at once (defaults: 32 vCPU, 128 GB), and it stops after 15 idle minutes. Job logs go to `s3://<bucket>/logs/` and expire after 30 days.
- **Ingestion on EMR** (`run 2025` without `--no-ingest`) needs internet egress. Give the application a VPC with a NAT gateway, or keep ingesting locally as above.
- **MLflow:** research commands log to `MLFLOW_TRACKING_URI`; export it before `run_job.sh` (for example, a SageMaker MLflow tracking server).
- **Tear down:** `terraform destroy`. The bucket must be emptied first; `force_destroy` is off on purpose.

## 3. Databricks

Prerequisites: a workspace, the Databricks CLI (≥ 0.240) authenticated to it, and an external location covering the bucket.

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev --var lakehouse_root=s3://<bucket>/lakehouse
databricks bundle run pitwall_season -t dev --params season=2025
```

The job runs `pitwall run <season>`, then `ingest openf1`, then the lap build, on a two-worker cluster. The schedule (Mondays 06:00 UTC, like Dagster's) is paused until you enable it. The package is installed as a wheel without the `spark` extra, so the runtime's Spark and Delta are used. `conf/` is synced with the bundle and found through `PITWALL_CONF_DIR`.

## What is and isn't verified

| Check | Status |
|---|---|
| Pipeline on S3 through `s3a://` (copy, rebuild, fingerprints) | ✅ Ran locally against SeaweedFS, see section 1 |
| `conf/*.yaml` profiles load as intended | ✅ `tests/test_config.py` |
| Terraform | ✅ `terraform validate` (AWS provider 5.x) |
| `run_job.sh` | ✅ shellcheck |
| `databricks.yml` | ✅ Validates against the Databricks CLI v0.240 bundle schema |
| EMR image | ✅ Builds from the public EMR 7.5 base. Inside it, `pitwall` runs under EMR's `spark-submit` (Spark 3.5.2-amzn, Delta 3.2.1-amzn) with the `aws` profile and `run_job.sh`'s Delta flags. A copied season's gold tables fingerprint identically to local (`deploy/aws/smoke_test.sh` → [`emr_image_check.json`](../reports/engineering/emr_image_check.json)) |
| IAM in a real account, job runs on EMR Serverless, Databricks deploy | ⬜ Needs an account; the commands are above |
