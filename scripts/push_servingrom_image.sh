#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-servingrom-tel-v1}
sudo docker save "$IMAGE" | podman load
podman push --tls-verify=false "$IMAGE"
curl -fsSI \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  "http://110.120.0.3:8889/v2/infra/qwen36-pd-worker/manifests/${IMAGE##*:}" \
  | tr -d '\r' | grep -i '^Docker-Content-Digest:'
