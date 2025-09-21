# Image NVIDIA officielle avec CUDA 12.4 + PyTorch
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Installer quelques dépendances système utiles
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /workspace

# Ajouter le repo dans le PYTHONPATH
ENV PYTHONPATH=/workspace

# Copier requirements
COPY docker/requirements.txt ./requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier ton code
COPY . .

# Commande par défaut (peut être écrasée avec docker run)
CMD ["python", "scripts/run_asr_pipeline.py", "tests/data/test.mp3", "--out", "outputs", "--lang", "fr", "--ns", "1"]