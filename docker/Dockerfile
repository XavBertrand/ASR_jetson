# Image NVIDIA officielle avec CUDA 12.4 + PyTorch
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Métadonnées de l'image
LABEL org.opencontainers.image.source="https://github.com/XavBertrand/ASR_jetson"
LABEL org.opencontainers.image.description="ASR Agent with GPU support for speech recognition"

# Éviter les interactions pendant l'installation
ENV DEBIAN_FRONTEND=noninteractive

# Installer les dépendances système + nvidia-utils pour nvidia-smi
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    gnupg \
    software-properties-common \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Définir le dossier de travail
WORKDIR /workspace

# Variables d'environnement pour NVIDIA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONPATH=/workspace

# Copier requirements en premier (pour le cache Docker)
COPY docker/requirements.txt ./requirements.txt

# Installer les dépendances Python de base
RUN pip install --no-cache-dir -r requirements.txt

# Installer torchaudio et torchvision avec CUDA 12.4
RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cu124

# Installer TensorRT et dépendances
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt

# Installer faster-whisper avec support TensorRT
RUN pip install --no-cache-dir faster-whisper[tensorrt]

# Installer CTranslate2 avec support TensorRT
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
    ctranslate2[tensorrt]

# Installer Nemo (compatible torch 2.4 de NVIDIA)
RUN pip install --no-cache-dir nemo_toolkit[asr]==2.0.0

# Copier le code
COPY . .

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