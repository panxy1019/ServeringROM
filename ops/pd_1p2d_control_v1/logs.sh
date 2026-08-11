#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/common.sh"

component=${1:-all}
lines=${2:-200}
pod=$(pod_name)
[[ -n "$pod" ]] || { echo "$DEPLOYMENT has no Pod" >&2; exit 2; }

case "$component" in
  pod) kube -n "$NAMESPACE" logs "$pod" --tail="$lines" ;;
  prefill|decode-a|decode-b|proxy|device-collector)
    kube -n "$NAMESPACE" exec "$pod" -- tail -n "$lines" "/var/log/qwen36-pd/$component.log"
    ;;
  all)
    for name in prefill decode-a decode-b proxy device-collector; do
      echo "===== $name ====="
      kube -n "$NAMESPACE" exec "$pod" -- sh -c "test -f /var/log/qwen36-pd/$name.log && tail -n $lines /var/log/qwen36-pd/$name.log || true"
    done
    ;;
  *) echo "Usage: $0 [all|pod|prefill|decode-a|decode-b|proxy|device-collector] [lines]" >&2; exit 2 ;;
esac
