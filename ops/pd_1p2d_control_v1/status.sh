#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/common.sh"

echo "== Deployment =="
kube -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
  -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas
echo "== Pod =="
kube -n "$NAMESPACE" get pod -l app="$DEPLOYMENT" \
  -o custom-columns=NAME:.metadata.name,PHASE:.status.phase,READY:.status.containerStatuses[0].ready,RESTARTS:.status.containerStatuses[0].restartCount,NODE:.spec.nodeName 2>/dev/null || true
echo "== Service endpoints =="
kube -n "$NAMESPACE" get endpoints "$SERVICE" -o wide 2>/dev/null || true

pod=$(pod_name)
if [[ -n "$pod" ]]; then
  echo "== In-container startup state =="
  kube -n "$NAMESPACE" exec "$pod" -- sh -c '
    test -f /var/run/qwen36-pd/READY && echo READY_FILE=present || echo READY_FILE=absent
    for file in npu-mapping.json service-device-map.txt effective-config.txt; do
      if [ -f "/var/run/qwen36-pd/$file" ]; then
        echo "--- $file ---"; cat "/var/run/qwen36-pd/$file"
      fi
    done
    curl --noproxy "*" -fsS --max-time 3 http://127.0.0.1:8080/health 2>/dev/null || true
    echo
  '
fi
