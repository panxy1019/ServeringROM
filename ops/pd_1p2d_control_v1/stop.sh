#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/common.sh"

kube -n "$NAMESPACE" scale deployment/"$DEPLOYMENT" --replicas=0
kube -n "$NAMESPACE" wait --for=delete pod -l app="$DEPLOYMENT" --timeout=20m || true
echo "Stopped $DEPLOYMENT. Deployment and persistent ServingROM results were retained."
"$ROOT/status.sh"
