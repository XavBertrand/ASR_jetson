# Image NVIDIA officielle avec CUDA 12.6 + PyTorch
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.09-py3
FROM ${PYTORCH_IMAGE}

# --- Build-time metadata (injectés par le workflow) ---
ARG GIT_BRANCH=unknown
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
ARG APP_VERSION=dev

# Métadonnées de l'image (alignées avec l'image Jetson)
LABEL org.opencontainers.image.source="https://github.com/XavBertrand/ASR_jetson" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.ref.name="${GIT_BRANCH}" \
      org.opencontainers.image.description="ASR Agent with GPU support for speech recognition"

# Variables runtime (inspectables dans le container)
ENV GIT_BRANCH="${GIT_BRANCH}" \
    GIT_SHA="${VCS_REF}" \
    APP_VERSION="${APP_VERSION}" \
    BUILD_DATE="${BUILD_DATE}" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

# Installer les dépendances système + nvidia-utils pour nvidia-smi
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    git \
    wget \
    curl \
    tini \
    libsndfile1 \
    nvidia-utils-535 \
    openjdk-21-jre-headless \
    && add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg6 \
    && apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Installation de uv + Python 3.11 pour aligner les dépendances avec Jetson
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && rm -rf /root/.cache
RUN uv python install 3.11 && uv venv --python 3.11 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/venv"

# Définir le dossier de travail
WORKDIR /workspace

# Variables d'environnement pour NVIDIA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONPATH=/workspace \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    PATH="/usr/lib/jvm/java-21-openjdk-amd64/bin:${PATH}" \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

# Préparer les métadonnées du projet pour profiter du cache Docker
COPY pyproject.toml README.md ./
COPY src ./src

# Utiliser l'index PyTorch CUDA 12.6 pour l'installation des dépendances
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126

# Installer les dépendances Python de base depuis le projet
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-cache-dir -e ".[media,tensorrt]" && \
    rm -rf /root/.cache

# Mettre à jour PyTorch et bibliothèques associées vers CUDA 12.6 (torch 2.8.0 GPU)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0+cu126 \
    torchvision==0.19.0+cu126 \
    torchaudio==2.8.0+cu126 && \
    rm -rf /root/.cache

# Pyannote > 4.0.1 pour compatibilité la plus récente
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-cache-dir --upgrade "pyannote-audio>4.0.1" && \
    rm -rf /root/.cache

# Installer TensorRT et dépendances
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt && \
    rm -rf /root/.cache

# Installer faster-whisper avec support TensorRT
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-cache-dir faster-whisper[tensorrt] && \
    rm -rf /root/.cache

# Installer CTranslate2 avec support TensorRT
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
    ctranslate2[tensorrt] && \
    rm -rf /root/.cache

# Copier le code
COPY . .

# Préparer les répertoires de cache alignés avec l'image Jetson
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch

# Script de vérification GPU
RUN echo '#!/bin/bash\n\
echo "=== Vérification GPU ==="\n\
echo "nvidia-smi:"\n\
nvidia-smi || echo "nvidia-smi non disponible"\n\
echo ""\n\
echo "PyTorch CUDA:"\n\
python -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\"); print(f\"Device count: {torch.cuda.device_count()}\"); print(f\"Device name: {torch.cuda.get_device_name() if torch.cuda.is_available() else \\\"N/A\\\"}\")"\n\
echo ""\n\
echo "faster-whisper GPU:"\n\
python -c "try:\n    from faster_whisper import WhisperModel\n    model = WhisperModel(\\\"tiny\\\", device=\\\"cuda\\\", compute_type=\\\"int8\\\")\n    print(\\\"✅ faster-whisper peut utiliser GPU\\\")\nexcept Exception as e:\n    print(f\\\"❌ Erreur faster-whisper GPU: {e}\\\")\n"\n\
' > /usr/local/bin/check-gpu && chmod +x /usr/local/bin/check-gpu

# Vérification initiale
RUN /usr/local/bin/check-gpu

# Créer le dossier outputs
RUN mkdir -p outputs

# Commande par défaut
CMD ["python", "scripts/run_asr_pipeline.py", "--help"]
