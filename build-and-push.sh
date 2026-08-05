#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730}
PROJECT_DIR=${PROJECT_DIR:-/home/admin/qwen36_pd_1p2d}
BUILD_LOG=${BUILD_LOG:-$PROJECT_DIR/markdowns/build-$(date -u +%Y%m%dT%H%M%SZ).log}

mkdir -p "$(dirname "$BUILD_LOG")"
echo "Building $IMAGE from $PROJECT_DIR"
sudo docker build --network=host --progress=plain -t "$IMAGE" "$PROJECT_DIR" 2>&1 | tee "$BUILD_LOG"
sudo docker image inspect "$IMAGE" --format '{{.Id}} {{.Size}}'

# The private registry is HTTP-only. Avoid changing Docker daemon-wide
# insecure-registry settings: transfer the built image to Podman and push with
# TLS verification explicitly disabled for this command only.
sudo docker save "$IMAGE" | podman load 2>&1 | tee -a "$BUILD_LOG"
podman push --tls-verify=false "$IMAGE" 2>&1 | tee -a "$BUILD_LOG"

curl -fsSI \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  "http://110.120.0.3:8889/v2/infra/qwen36-pd-worker/manifests/${IMAGE##*:}" \
  | tr -d '\r' | grep -i '^Docker-Content-Digest:' | tee -a "$BUILD_LOG"
