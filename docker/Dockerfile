# Étape 1 : partir d'une image PyTorch NVIDIA compatible (PC local, x86_64)
# Pour Jetson, on remplacera par nvcr.io/nvidia/l4t-pytorch:<version>
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Définir le dossier de travail
WORKDIR /app

# Installer dépendances Python
COPY docker/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier ton code source
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Créer dossier pour sorties
RUN mkdir -p outputs

# Entrypoint par défaut (on peut override avec docker run ...)
ENTRYPOINT ["python", "src/pipeline.py"]
