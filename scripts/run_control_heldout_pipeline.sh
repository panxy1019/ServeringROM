#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
RUN_ROOT=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 "$REPO_ROOT/scripts/build_proxy_lifecycle.py" "$RUN_ROOT"
python3 "$REPO_ROOT/scripts/build_internal_telemetry.py" "$RUN_ROOT"
python3 "$REPO_ROOT/scripts/build_full_order_snapshots.py" "$RUN_ROOT" --period-ms 200
python3 "$REPO_ROOT/scripts/validate_proxy_lifecycle.py" "$RUN_ROOT"
python3 "$REPO_ROOT/scripts/validate_internal_telemetry.py" "$RUN_ROOT"
python3 "$REPO_ROOT/scripts/validate_full_order_snapshots.py" "$RUN_ROOT"
python3 "$REPO_ROOT/scripts/build_control_snapshots.py" "$RUN_ROOT"
python3 "$REPO_ROOT/scripts/validate_control_heldout_run.py" "$RUN_ROOT"
python3 "$REPO_ROOT/scripts/seal_servingrom_run.py" "$RUN_ROOT"
