#!/usr/bin/env bash
# Usage:
#   ./docker/build.sh local      # build amd64 only, --load
#   ./docker/build.sh multiarch  # build amd64+arm64, --push

set -euo pipefail

MODE="${1:-local}"
IMAGE_NAME="${IMAGE_NAME:-xavier/asr-agent}"
TAG="${TAG:-dev}"
WITH_NEMO="${WITH_NEMO:-1}"

echo "Building image ${IMAGE_NAME}:${TAG} (mode: ${MODE})..."

# Ensure builder
if docker buildx inspect asr-builder >/dev/null 2>&1; then
  docker buildx use asr-builder
else
  docker buildx create --name asr-builder --use
fi

BUILD_ARGS=()
if [ "$WITH_NEMO" = "1" ]; then
  BUILD_ARGS+=(--build-arg WITH_NEMO=1)
  echo "NeMo/TitaNet-S will be installed."
else
  echo "NeMo/TitaNet-S disabled."
fi

DOCKERFILE="docker/Dockerfile"

if [ "$MODE" = "local" ]; then
  # IMPORTANT: on force TARGETARCH=amd64 pour le build local
  docker buildx build \
    --builder asr-builder \
    --platform linux/amd64 \
    --build-arg TARGETARCH=amd64 \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${DOCKERFILE}" \
    --load \
    .
  echo "✅ Local image loaded: ${IMAGE_NAME}:${TAG}"

elif [ "$MODE" = "multiarch" ]; then
  # Laisser buildx injecter automatiquement TARGETARCH pour chaque plate-forme
  docker buildx build \
    --builder asr-builder \
    --platform linux/amd64,linux/arm64 \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${DOCKERFILE}" \
    --push \
    .
  echo "✅ Multi-arch manifest pushed: ${IMAGE_NAME}:${TAG}"

else
  echo "❌ Unknown mode: $MODE (use 'local' or 'multiarch')" >&2
  exit 1
fi
