#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 results/<experiment_id>/<run_id>" >&2
  exit 2
fi

RUN_ROOT=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python "${REPO_ROOT}/scripts/build_proxy_lifecycle.py" "${RUN_ROOT}"
python "${REPO_ROOT}/scripts/build_internal_telemetry.py" "${RUN_ROOT}"
python "${REPO_ROOT}/scripts/build_full_order_snapshots.py" "${RUN_ROOT}" --period-ms 200
python "${REPO_ROOT}/scripts/validate_proxy_lifecycle.py" "${RUN_ROOT}"
python "${REPO_ROOT}/scripts/validate_internal_telemetry.py" "${RUN_ROOT}"
python "${REPO_ROOT}/scripts/validate_full_order_snapshots.py" "${RUN_ROOT}"
python "${REPO_ROOT}/scripts/seal_servingrom_run.py" "${RUN_ROOT}"

