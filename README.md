# Lightweight ASR Pipeline with Diarization

This repository provides a **lightweight, modular, and efficient Automatic Speech Recognition (ASR) pipeline** designed to run locally on both desktop GPUs and edge devices such as the **Jetson Orin Nano**.
It combines optional noise suppression, speaker diarization, transcription, and post-processing in a single, end-to-end workflow driven by Pyannote + Faster-Whisper.

---

## ✨ Features

* 🎧 **Denoising (optional)** via [RNNoise](https://github.com/xiph/rnnoise).
* 👥 **Speaker diarization** fully powered by [Pyannote Audio](https://github.com/pyannote/pyannote-audio).
* 📝 **Automatic Speech Recognition (ASR)** using [FasterWhisper](https://github.com/SYSTRAN/faster-whisper).
* ⚡ **Optimized for Jetson Orin Nano**: designed to run locally with CUDA/TensorRT acceleration.
* 🧱 **uv + pyproject.toml** build system (no `requirements.txt` needed).
* 🥪 Unit and integration tests with pytest.

---

## 📂 Repository Structure

```
ASR_jetson/
├── src/
│   └── asr_jetson/
│       ├── preprocessing/        # RNNoise wrapper + audio conversion helpers
│       ├── diarization/          # Pyannote-based diarization pipeline
│       ├── asr/                  # FasterWhisper helpers
│       ├── postprocessing/       # Text cleaning, anonymisation, and report generation
│       ├── pipeline/             # End-to-end pipeline orchestration (core + CLI)
│       └── utils/                # Configs, logging helpers
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
* CUDA-enabled GPU (recommended for realtime / large models)
* [ffmpeg](https://ffmpeg.org/) in PATH (required for audio conversion + RNNoise)
* [uv](https://github.com/astral-sh/uv) installed (`pip install uv`)
* (For diarization) a valid Hugging Face token stored in `HUGGINGFACE_TOKEN`

### Setup

```bash
git clone https://github.com/XavBertrand/ASR_jetson.git
cd ASR_jetson

# Install dependencies (desktop Linux / WSL with CUDA 12.4 wheels)
uv sync --extra dev --extra media --extra gpu-linux

# Jetson (aarch64) simply omits the gpu-linux extra:
# uv sync --extra dev --extra media --extra gpu-jetson

# Authenticate with Hugging Face once for Pyannote access
export HUGGINGFACE_TOKEN=hf_xxx
```

---

## ▶️ Usage

### Run from CLI

```bash
uv run asr-pipeline \
  --audio path/to/file.wav \
  --out-dir outputs \
  --speakers 2 \
  --pyannote-pipeline pyannote/speaker-diarization-3.1
```

Or directly:

```bash
uv run python -m asr_jetson --audio path/to/file.wav

# Or pass a token explicitly (useful in CI)
uv run asr-pipeline --audio file.wav --pyannote-token "$HUGGINGFACE_TOKEN"
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

To skip GPU tests (or when Pyannote cannot run):

```bash
uv run pytest -m "not gpu"
```

Integration tests rely on Pyannote and may require downloading weights from Hugging Face; set
`HUGGINGFACE_TOKEN` accordingly or mark the `integration` tests to skip.

---

## 📊 Benchmarks

| Model                | Device             | 1h audio runtime |
| -------------------- | ------------------ | ---------------- |
| FasterWhisper-Large  | Desktop GPU (4070) | ~12 min          |
| FasterWhisper-Medium | Jetson Orin Nano   | ~25–30 min       |

*(Approximate values; depends on compute type and GPU clocks)*

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

* [RNNoise](https://github.com/xiph/rnnoise)
* [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
* [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
* [uv](https://github.com/astral-sh/uv) for the packaging workflow
* [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
* [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
* [Whisper & FasterWhisper](https://github.com/openai/whisper)
* [RNNoise](https://github.com/xiph/rnnoise)
* [uv](https://github.com/astral-sh/uv)
