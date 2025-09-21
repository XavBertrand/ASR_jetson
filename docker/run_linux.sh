#!/usr/bin/env bash
# Run the ASR Agent container with GPU by default.
# - On amd64 (PC): uses --gpus all
# - On aarch64 (Jetson): uses --runtime=nvidia

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-xavier/asr-agent}"
TAG="${TAG:-latest}"

# Mounts
HOST_PWD="$(pwd)"
MOUNT_APP="${MOUNT_APP:-$HOST_PWD}"

# Detect arch
ARCH="$(uname -m || true)"
GPU_FLAG=""
case "$ARCH" in
  x86_64|amd64)
    GPU_FLAG="--gpus all"
    ;;
  aarch64|arm64)
    GPU_FLAG="--runtime=nvidia"
    ;;
  *)
    echo "⚠️  Arch '$ARCH' not recognized. Defaulting to CPU run (no GPU flag)."
    GPU_FLAG=""
    ;;
esac

# Optional: forward a local cache dir for HF models if you want persistent weights
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# Container name (optional)
NAME="${NAME:-asr-agent}"

echo "Running ${IMAGE_NAME}:${TAG}"
echo "Detected arch: ${ARCH} | Using GPU flag: '${GPU_FLAG}'"
echo "Mount: ${MOUNT_APP} -> /app"

exec docker run --rm -it \
  ${GPU_FLAG} \
  --name "${NAME}" \
  -v "${MOUNT_APP}:/app" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -w /app \
  "${IMAGE_NAME}:${TAG}" \
  bash
