#!/usr/bin/env bash
set -euo pipefail

KUBECTL=${KUBECTL:-/usr/local/bin/k3s kubectl}
PROJECT_DIR=${PROJECT_DIR:-/home/admin/testpanxy/infralearning/qwen36_pd_1p2d}

sudo $KUBECTL -n infra-learning create configmap qwen36-pd-worker-scripts \
  --from-file=discover_npu_mapping.py="$PROJECT_DIR/scripts/discover_npu_mapping.py" \
  --from-file=check_target_npus_idle.py="$PROJECT_DIR/scripts/check_target_npus_idle.py" \
  --from-file=pd-worker-entrypoint.sh="$PROJECT_DIR/scripts/pd-worker-entrypoint.sh" \
  --from-file=pd_proxy.py="$PROJECT_DIR/scripts/pd_proxy.py" \
  --from-file=test_pd_proxy_scheduler.py="$PROJECT_DIR/scripts/test_pd_proxy_scheduler.py" \
  --from-file=mixed_pd_load.py="$PROJECT_DIR/scripts/mixed_pd_load.py" \
  --from-file=status.sh="$PROJECT_DIR/scripts/status.sh" \
  --from-file=smoke.sh="$PROJECT_DIR/scripts/smoke.sh" \
  --dry-run=client -o yaml | sudo $KUBECTL apply -f -
sudo $KUBECTL apply -f "$PROJECT_DIR/k8s/qwen36-pd-worker.yaml"
sudo $KUBECTL -n infra-learning get deployment ray-vllm-pd-worker-qwen36-27b
sudo $KUBECTL -n infra-learning get service qwen36-pd
echo "Workload registered at replicas=0. Run start.sh only after the A3 idle check passes."
