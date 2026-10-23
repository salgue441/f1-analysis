"""Rebuild one season on S3 from copied bronze and compare it with the local lakehouse.

Run inside the workbench with the `s3` compose profile up (`make s3-verify`):
copy bronze -> rebuild silver + gold on s3a:// -> fingerprint every season table in both
lakehouses -> write reports/engineering/cloud_verification.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "engineering" / "cloud_verification.json"
LOCAL_ROOT = "/workspace/data/lakehouse"
SKIP = ["openf1_car_data", "openf1_location"]  # telemetry: large, and not needed to verify


def pitwall(env: str, *args: str) -> str:
    result = subprocess.run(
        ["pitwall", *args],
        env={**os.environ, "PITWALL_ENV": env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def json_tail(text: str) -> dict:
    return json.loads(text[text.index("{") :])


def main(season: int) -> int:
    started = time.perf_counter()
    exclude = [x for name in SKIP for x in ("--exclude", name)]
    copied = json_tail(
        pitwall("s3local", "copy", "--from", LOCAL_ROOT, "--season", str(season), *exclude)
    )
    copy_s = time.perf_counter() - started
    pitwall("s3local", "run", str(season), "--no-ingest")
    pitwall("s3local", "run", str(season), "--no-ingest", "--laps")
    rebuild_s = time.perf_counter() - started - copy_s

    s3 = json_tail(pitwall("s3local", "fingerprints", "--season", str(season)))
    local = json_tail(pitwall("local", "fingerprints", "--season", str(season)))
    common = sorted(set(s3) & set(local))
    differ = [t for t in common if s3[t] != local[t]]
    result = {
        "season": season,
        "store": "SeaweedFS S3 API via s3a:// (conf/s3local.yaml)",
        "bronze_tables_copied": len(copied),
        "season_tables_compared": len(common),
        "identical": len(common) - len(differ),
        "differ": {t: {"s3": s3[t], "local": local[t]} for t in differ},
        "only_on_s3": sorted(set(s3) - set(local)),
        "only_local": sorted(set(local) - set(s3)),
        "copy_s": round(copy_s, 1),
        "rebuild_s": round(rebuild_s, 1),
        "rows_compared": sum(s3[t]["rows"] for t in common),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not differ and common else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024))
