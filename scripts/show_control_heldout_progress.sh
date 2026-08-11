#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST=${1:-$ROOT/.campaign/servingrom-control-heldout-v1/CONTROL_HELDOUT_MANIFEST.json}
python3 - "$MANIFEST" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"benchmark={d['benchmark_id']} status={d['status']} updated={d['updated_at']}")
print("counts=",d.get("counts"))
for row in d["runs"]:
 if row["status"] in ("RUNNING","FAILED"): print(row["status"],row["plan_id"],row.get("run_id"),row.get("error"))
PY
