#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST=${1:-$REPO_ROOT/.campaign/servingrom-control-dataset-v1/dataset_run_manifest.json}
python3 - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
print(f"dataset={d['dataset_id']} status={d['status']} updated={d['updated_at']}")
print("counts=", d.get("counts"))
for r in d["runs"]:
    if r["status"] in ("RUNNING","FAILED"):
        print(r["status"], r["plan_id"], r.get("run_id"), r.get("started_at"), r.get("error"))
PY
