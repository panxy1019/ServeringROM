#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/admin/qwen36_pd_1p2d}
IMAGE=${IMAGE:-110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-servingrom-tel-v1}
BASE_IMAGE=${BASE_IMAGE:-110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730}
BASE_MANIFEST_DIGEST=${BASE_MANIFEST_DIGEST:-sha256:15c3a3db3772807cc09d9ad37756cd973e3b078956b6a239375ec4ae23317133}
BASE_CONFIG_DIGEST=${BASE_CONFIG_DIGEST:-sha256:6d454e6d5715ac8792868408e57f9287aa0444867db23747b797ddaae5ff924a}
REPOSITORY_COMMIT=${REPOSITORY_COMMIT:-$(git -C "$PROJECT_DIR" rev-parse HEAD)}

actual_config_digest=$(sudo docker image inspect "$BASE_IMAGE" --format '{{.Id}}')
if [[ "$actual_config_digest" != "$BASE_CONFIG_DIGEST" ]]; then
  echo "Base image config digest mismatch: $actual_config_digest" >&2
  exit 1
fi

sudo docker build \
  --network=host \
  --progress=plain \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "BASE_MANIFEST_DIGEST=$BASE_MANIFEST_DIGEST" \
  --build-arg "REPOSITORY_COMMIT=$REPOSITORY_COMMIT" \
  -f "$PROJECT_DIR/docker/Dockerfile.servingrom" \
  -t "$IMAGE" \
  "$PROJECT_DIR"

sudo docker run --rm \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  --entrypoint python "$IMAGE" \
  /opt/servingrom/tests/test_engine_telemetry_patch.py

sudo docker image inspect "$IMAGE" --format '{{.Id}} {{.Size}}'
