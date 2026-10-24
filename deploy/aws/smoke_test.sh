#!/usr/bin/env bash
# Run pitwall inside the EMR Serverless image, locally, with EMR's own spark-submit, Spark and
# Delta, the `aws` profile and the same Delta flags as run_job.sh. Copies one season of gold
# from the local lakehouse, fingerprints it, and compares with the workbench's fingerprints.
#   docker build -f deploy/aws/Dockerfile -t pitwall-emr:check .
#   deploy/aws/smoke_test.sh [season]      -> reports/engineering/emr_image_check.json
set -euo pipefail

season="${1:-2024}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
out="$repo/data/emr_check"
mkdir -p "$out"
cd "$repo"

docker run --rm \
  -v "$repo/data/lakehouse:/data/lakehouse:ro" \
  -v "$repo/deploy/aws/pitwall_job.py:/job/pitwall_job.py:ro" \
  -v "$out:/out" \
  -e PITWALL_LAKEHOUSE_ROOT=/tmp/lh \
  --entrypoint sh pitwall-emr:check -c '
    d=/usr/share/aws/delta/lib
    submit() {
      /usr/lib/spark/bin/spark-submit --master "local[4]" \
        --conf spark.jars=$d/delta-spark.jar,$d/delta-storage.jar \
        --conf spark.submit.pyFiles=$d/delta-spark.jar \
        --conf spark.pyspark.python=/usr/bin/python3.11 \
        --conf spark.pyspark.driver.python=/usr/bin/python3.11 \
        /job/pitwall_job.py "$@" 2>>/out/spark.log
    }
    /usr/lib/spark/bin/spark-submit --version 2>&1 | grep -m1 version > /out/spark_version.txt
    submit copy --from /data/lakehouse --layer gold --season "$0" > /out/copy.txt
    submit fingerprints --season "$0" > /out/fingerprints.txt
  ' "$season"

docker compose exec -T workbench \
  pitwall fingerprints --season "$season" > "$out/local.txt"

docker compose exec -T workbench python - "$season" <<'EOF'
import json, sys
from pathlib import Path

def tail_json(path):
    text = Path(path).read_text()
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]

out = Path("/workspace/data/emr_check")
emr = tail_json(out / "fingerprints.txt")
local = tail_json(out / "local.txt")
common = sorted(t for t in emr if t in local)
result = {
    "season": int(sys.argv[1]),
    "image": "deploy/aws/Dockerfile (EMR Serverless emr-7.5.0 base)",
    "spark": (out / "spark_version.txt").read_text().split("version")[-1].strip(),
    "profile": "aws (spark.master empty, delta_pip false, EMR Delta jars)",
    "tables_compared": len(common),
    "identical": sum(emr[t] == local[t] for t in common),
    "rows_compared": sum(emr[t]["rows"] for t in common),
    "differ": [t for t in common if emr[t] != local[t]],
}
target = Path("/workspace/reports/engineering/emr_image_check.json")
target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
EOF
