#!/usr/bin/env bash
# Build multi-arch docker image for ASR Agent with NeMo/TitaNet-S

set -e

IMAGE_NAME="xavier/asr-agent"
TAG="latest"

echo "Building multi-arch image ($IMAGE_NAME:$TAG) with NeMo (TitaNet-S)..."

docker buildx create --use --name asr-builder 2>/dev/null || true

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --build-arg WITH_NEMO=1 \
  -t ${IMAGE_NAME}:${TAG} \
  -f docker/Dockerfile \
  .
