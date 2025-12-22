# Image NVIDIA officielle avec CUDA 12.6 + PyTorch
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.09-py3
FROM ${PYTORCH_IMAGE}
ARG ENABLE_TENSORRT=0

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
ENV PYTHONPATH=/workspace:/workspace/src \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    PATH="/usr/lib/jvm/java-21-openjdk-amd64/bin:${PATH}" \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

# Préparer les métadonnées du projet pour profiter du cache Docker
COPY pyproject.toml README.md uv.lock ./
COPY src ./src

# Utiliser l'index PyTorch CUDA 12.6 pour l'installation des dépendances
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126

# Installer les dépendances Python (une seule couche) et purger les artefacts volumineux
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eux; \
    uv sync --frozen --extra media; \
    uv pip install --no-cache-dir -e . --no-deps; \
    if [ "${ENABLE_TENSORRT}" = "1" ]; then \
        uv pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt; \
        uv pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 ctranslate2[tensorrt]; \
        uv pip install --no-cache-dir faster-whisper[tensorrt]; \
    fi; \
    PRECOMPILED_CUDNN="/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9"; \
    if [ -f "${PRECOMPILED_CUDNN}" ]; then \
        echo "Removing precompiled cuDNN engines to save space"; \
        rm -f "${PRECOMPILED_CUDNN}"; \
    fi; \
    find /opt/venv -type d -name "tests" -prune -exec rm -rf {} +; \
    find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} +; \
    find /opt/venv -name "*.pyc" -delete; \
    rm -rf /tmp/* /var/tmp/*

# Copier le code
COPY . .

# Préparer les répertoires de cache alignés avec l'image Jetson
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch

# Script de vérification GPU
RUN cat > /usr/local/bin/check-gpu <<'EOF' && chmod +x /usr/local/bin/check-gpu
#!/bin/bash
echo "=== Vérification GPU ==="
echo "nvidia-smi:"
nvidia-smi || echo "nvidia-smi non disponible"
echo ""
echo "PyTorch CUDA:"
python - <<'PYCODE'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")
PYCODE
echo ""
echo "faster-whisper GPU:"
python - <<'PYCODE'
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cuda", compute_type="int8")
    print("✅ faster-whisper peut utiliser GPU")
except Exception as e:
    print(f"❌ Erreur faster-whisper GPU: {e}")
PYCODE
EOF

# Vérification initiale
RUN /usr/local/bin/check-gpu

# Créer le dossier outputs
RUN mkdir -p outputs

# Commande par défaut
CMD ["python", "scripts/run_asr_pipeline.py", "--help"]
