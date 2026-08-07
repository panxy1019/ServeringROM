#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec python3 "$ROOT/scripts/run_rom_modeling.py" --dataset-root "$1" --index-dir "$2" --output-root "$3" --config "$ROOT/configs/servingrom_rom_model_v1.json" --stop-after step11
