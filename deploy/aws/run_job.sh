#!/usr/bin/env bash
# Submit a pitwall command to EMR Serverless and print the job run id.
#   deploy/aws/run_job.sh run 2025 --no-ingest --laps
#   deploy/aws/run_job.sh fingerprints --season 2025
# Needs: AWS CLI v2 with credentials, `terraform apply` done in deploy/aws/terraform,
# and the image pushed (docs/cloud.md). Costs money while the job runs.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <pitwall command...>" >&2
  exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
tf() { terraform -chdir="$here/terraform" output -raw "$1"; }
app_id="$(tf application_id)"
role_arn="$(tf job_role_arn)"
bucket="$(tf bucket)"
root="$(tf lakehouse_root)"

aws s3 cp "$here/pitwall_job.py" "s3://$bucket/code/pitwall_job.py"

args_json="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")"
delta=/usr/share/aws/delta/lib
env_conf=""
for var in PITWALL_ENV=aws "PITWALL_LAKEHOUSE_ROOT=$root" ${MLFLOW_TRACKING_URI:+"MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"}; do
  env_conf+=" --conf spark.emr-serverless.driverEnv.$var --conf spark.executorEnv.$var"
done
# The Delta jar also carries its Python API (`delta.tables`), hence pyFiles.
submit="--conf spark.jars=$delta/delta-spark.jar,$delta/delta-storage.jar"
submit+=" --conf spark.submit.pyFiles=$delta/delta-spark.jar$env_conf"

job_driver="$(python3 -c 'import json, sys; print(json.dumps({"sparkSubmit": {
    "entryPoint": sys.argv[1], "entryPointArguments": json.loads(sys.argv[2]),
    "sparkSubmitParameters": sys.argv[3]}}))' \
  "s3://$bucket/code/pitwall_job.py" "$args_json" "$submit")"
overrides="$(python3 -c 'import json, sys; print(json.dumps({"monitoringConfiguration": {
    "s3MonitoringConfiguration": {"logUri": sys.argv[1]}}}))' "s3://$bucket/logs/")"

aws emr-serverless start-job-run \
  --application-id "$app_id" \
  --execution-role-arn "$role_arn" \
  --name "pitwall $*" \
  --job-driver "$job_driver" \
  --configuration-overrides "$overrides" \
  --query jobRunId --output text
