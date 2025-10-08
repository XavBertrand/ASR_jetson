# Lightweight ASR Pipeline with Diarization

This repository provides a **lightweight, modular, and efficient Automatic Speech Recognition (ASR) pipeline** designed to run locally on both desktop GPUs and edge devices such as the **Jetson Orin Nano**.
It combines noise suppression, Voice Activity Detection (VAD), speaker diarization, and ASR transcription in a single, end-to-end workflow.

---

## ✨ Features

* 🎧 **Noise suppression** with [RNNoise](https://github.com/xiph/rnnoise) to enhance speech quality.
* 🎤 **Voice Activity Detection (VAD)** using [Silero VAD](https://github.com/snakers4/silero-vad) for accurate speech segmentation.
* 👥 **Speaker diarization** with **TitaNet-S embeddings** and **spectral clustering**, enabling speaker-attributed transcriptions.
* 📝 **Automatic Speech Recognition (ASR)** using [FasterWhisper](https://github.com/SYSTRAN/faster-whisper) or **NVIDIA FastConformer** (via [NeMo](https://github.com/NVIDIA/NeMo)).
* ⚡ **Optimized for Jetson Orin Nano**: runs locally with CUDA/TensorRT acceleration.
* 🧱 **uv + pyproject.toml** build system (no `requirements.txt` needed).
* 🥪 Unit and integration tests with pytest.

---

## 📂 Repository Structure

```
ASR_jetson/
├── src/
│   └── asr_jetson/
│       ├── preprocessing/        # RNNoise wrapper
│       ├── vad/                  # Silero VAD integration
│       ├── diarization/          # TitaNet-S embeddings + clustering
│       ├── asr/                  # FasterWhisper / NeMo FastConformer
│       ├── postprocessing/       # Text cleaning and formatting
│       ├── pipeline/             # End-to-end pipeline orchestration (core + CLI)
│       ├── io/                   # Audio I/O and storage utilities
│       └── utils/                # Configs, helpers
│
├── configs/                      # (optional) runtime YAML configs
│   ├── dev.yaml
│   └── jetson.yaml
│
├── tests/                        # Unit & integration tests
│   ├── test_pipeline.py
│   └── data/
│
├── docker/
│   ├── Dockerfile                # Multi-arch (x86_64 + ARM64) build
│   └── Dockerfile.jetson         # Jetson Orin Nano deployment
│
├── pyproject.toml                # Dependencies, scripts, and settings
├── uv.lock                       # uv dependency lock file
└── README.md
```

---

## 🛠 Installation (with uv)

### Prerequisites

* Python ≥ 3.10
* CUDA-enabled GPU (recommended)
* [ffmpeg](https://ffmpeg.org/) in PATH
* [uv](https://github.com/astral-sh/uv) installed (`pip install uv`)

### Setup

```bash
git clone https://github.com/XavBertrand/ASR_jetson.git
cd ASR_jetson

# Create virtual environment and install dependencies
uv sync --extra dev --extra media

# (Optional) add GPU support on Windows
uv add "torch==2.4.0+cu124" --extra-index-url https://download.pytorch.org/whl/cu124
```

---

## ▶️ Usage

### Run from CLI

```bash
uv run asr-pipeline --audio path/to/file.wav --out out/transcript.json
```

Or directly:

```bash
uv run python -m asr_jetson --audio path/to/file.wav
```

### Example Output

```json
[
  {
    "speaker": "SPEAKER_1",
    "start": 0.5,
    "end": 3.2,
    "text": "Hello everyone, thanks for joining the meeting today."
  },
  {
    "speaker": "SPEAKER_2",
    "start": 3.3,
    "end": 5.7,
    "text": "Good morning, let's get started."
  }
]
```

---

## 🐳 Docker

### 🧹 Local / Desktop build (x86_64)

```bash
docker build -t asr-jetson:dev -f docker/Dockerfile .
```

### 🚀 Jetson Orin Nano build

Uses NVIDIA’s `l4t-ml` base (includes CUDA + PyTorch).
Make sure JetPack ≥ 6.0.

```bash
docker build -t asr-jetson:jetson -f docker/Dockerfile.jetson .
```

**Key points:**

* No more `requirements.txt` — dependencies are installed via `uv sync` using `pyproject.toml`.
* Torch is already included in `l4t-ml`.
* Volumes can be mounted for I/O:

  ```bash
  docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/models:/models -v $(pwd)/output:/output asr-jetson:jetson
  ```

### 🔱 Multi-arch build (x86_64 + ARM64)

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t xavbertrand/asr-jetson:latest \
  -f docker/Dockerfile.jetson \
  --push .
```

---

## ✅ Testing

```bash
uv run pytest
```

To skip GPU tests on CPU:

```bash
pytest -m "not gpu"
```

---

## 📊 Benchmarks

| Model             | Device             | 1h audio runtime |
| ----------------- | ------------------ | ---------------- |
| Whisper Large     | Desktop GPU (4070) | ~12 min          |
| FasterWhisper-M   | Jetson Orin Nano   | ~25–30 min       |
| FastConformer-CTC | Jetson Orin Nano   | ~20–25 min       |

*(Approximate values depending on model and precision settings)*

---

## 🖊 Roadmap

* [ ] Add support for **online streaming transcription**
* [ ] Integrate **FastAPI service** for remote inference
* [ ] Add **speaker adaptation** (personalized profiles)
* [ ] Extend diarization with **overlap detection**

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* [Silero VAD](https://github.com/snakers4/silero-vad)
* [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
* [TitaNet](https://arxiv.org/abs/2110.04410)
* [Whisper & FasterWhisper](https://github.com/openai/whisper)
* [RNNoise](https://github.com/xiph/rnnoise)
* [uv](https://github.com/astral-sh/uv)
