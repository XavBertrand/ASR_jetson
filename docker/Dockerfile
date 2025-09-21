FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Installe wheels Torch CUDA correspondants (+cu124) puis le reste
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt

COPY . /app
CMD ["python3", "-m", "src.asr_agent"]
