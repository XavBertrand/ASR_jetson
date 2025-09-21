# syntax=docker/dockerfile:1.7-labs
################################################################################
# Multi-arch Dockerfile for:
# - linux/amd64 (Windows host via WSL2): CUDA runtime + PyTorch CUDA + Faster-Whisper GPU
# - linux/arm64/v8 (Jetson Orin/Nano): JetPack/Jetson base with TensorRT + PyTorch from L4T
#
# Build (multi-arch manifest):
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     --build-arg WITH_NEMO=1 \        # optionnel: installe NeMo/TitaNet-S
#     -t yourrepo/asr-agent:latest -f docker/Dockerfile .
#
# Run (x86 GPU):
#   docker run --rm -it --gpus all -v $PWD:/app yourrepo/asr-agent:latest
#
# Run (Jetson):
#   docker run --rm -it --runtime=nvidia -v $PWD:/app yourrepo/asr-agent:latest
################################################################################

ARG BASE_AMD64=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# JetPack 6.x (L4T r36.x) ML stack inclut PyTorch + TensorRT + CUDA pour Jetson (arm64)
ARG BASE_ARM64=nvcr.io/nvidia/l4t-ml:r36.2.0-py3

############################
# Stage: base for amd64
############################
FROM --platform=linux/amd64 ${BASE_AMD64} AS base_amd64

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA-enabled PyTorch stack (x86) – adapte si tu veux une autre version CUDA
ARG PYTORCH_CUDA=124
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install --index-url https://download.pytorch.org/whl/cu${PYTORCH_CUDA} \
        torch torchvision torchaudio

############################
# Stage: base for arm64 (Jetson)
############################
FROM --platform=linux/arm64 ${BASE_ARM64} AS base_arm64

# L4T-ML fournit déjà Python, PyTorch (Jetson), TensorRT, CUDA drivers userspace
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1

# Paquets système courants
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    git curl ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

############################
# Stage: common runtime (selected by TARGETARCH)
############################
ARG TARGETARCH
# Trick to select the right base stage per-arch
ARG BASE_STAGE=base_${TARGETARCH}
FROM ${BASE_STAGE} AS runtime

WORKDIR /app

# Copie le minimum d’abord pour profiter du cache pip
COPY docker/requirements.txt /app/docker/requirements.txt

# Dépendances Python (hors torch/torchaudio déjà gérés par les bases)
RUN python3 -m pip install -r /app/docker/requirements.txt

# Optionnel: installer NeMo pour TitaNet-S (diarisation). Très lourd: active seulement si nécessaire.
# Sur Jetson, c’est supporté mais peut rallonger la build; sur amd64, OK.
# ARG WITH_NEMO=0
# RUN if [ "$WITH_NEMO" = "1" ]; then \
#       python3 -m pip install 'nemo_toolkit[asr]==2.4.0' 'pytorch-lightning<2.4'; \
#     fi
RUN python3 -m pip install 'nemo_toolkit[asr]==2.4.0' 'pytorch-lightning<2.4'

# Astuces perf CTranslate2 + CUDA (faster-whisper)
# - Tu peux ajuster compute_type via ton code: int8 / int8_float16 / float16
ENV CT2_VERBOSE=0

# Copie du reste du projet
COPY . /app

# Vérifs légères (désactive si tu préfères un build plus silencieux)
# - Confirme torch + GPU côté x86 / Jetson
RUN python3 - <<'PY'
import torch, os
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
PY

# Entrypoint: tu peux remplacer par ton script ou garder juste le shell pour dev
# Exemples d’utilisation:
#   docker run ... python3 scripts/run_asr_pipeline.py --audio inputs/foo.wav
ENTRYPOINT ["/bin/bash"]
