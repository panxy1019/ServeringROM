#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi
RUN_ROOT=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

bash "$REPO_ROOT/scripts/run_snapshot_phase_a.sh" "$RUN_ROOT"
python3 "$REPO_ROOT/scripts/build_control_snapshots.py" "$RUN_ROOT"
python3 "$REPO_ROOT/scripts/validate_control_pilot_run.py" "$RUN_ROOT"
