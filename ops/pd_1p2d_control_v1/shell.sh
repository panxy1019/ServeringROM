#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/common.sh"
pod=$(pod_name)
[[ -n "$pod" ]] || { echo "$DEPLOYMENT has no Pod" >&2; exit 2; }
kube -n "$NAMESPACE" exec -it "$pod" -- /bin/bash
