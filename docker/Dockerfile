# syntax=docker/dockerfile:1.7-labs

################################################################################
# Bases par architecture
################################################################################
ARG BASE_AMD64=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG BASE_ARM64=nvcr.io/nvidia/l4t-ml:r36.2.0-py3
# On déclare TARGETARCH ici (et on le redéclarera avant le FROM final)
ARG TARGETARCH

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

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA (x86) – adapte la série CUDA si besoin
ARG PYTORCH_CUDA=124
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install --index-url https://download.pytorch.org/whl/cu${PYTORCH_CUDA} \
        torch torchvision torchaudio

############################
# Stage: base for arm64 (Jetson)
############################
FROM --platform=linux/arm64 ${BASE_ARM64} AS base_arm64

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    git curl ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

############################
# Stage runtime commun (sélection par arch)
############################
# IMPORTANT: redéclarer TARGETARCH juste avant ce FROM
ARG TARGETARCH
FROM base_${TARGETARCH} AS runtime

WORKDIR /app

# Déps Python (hors torch/torchaudio déjà gérés par les bases)
COPY docker/requirements.txt /app/docker/requirements.txt
RUN python3 -m pip install -r /app/docker/requirements.txt

# Optionnel: NeMo/TitaNet-S si demandé (lourd)
ARG WITH_NEMO=0
RUN if [ "$WITH_NEMO" = "1" ]; then \
      python3 -m pip install 'nemo_toolkit[asr]==2.4.0' 'pytorch-lightning<2.4'; \
    fi

ENV CT2_VERBOSE=0

# Copie du projet
COPY . /app

# Sanity check CUDA
RUN python3 - <<'PY'
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY

ENTRYPOINT ["/bin/bash"]
