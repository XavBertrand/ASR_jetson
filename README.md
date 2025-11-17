# Lightweight ASR Pipeline with Diarization

This repository provides a **lightweight, modular, and efficient Automatic Speech Recognition (ASR) pipeline** designed to run locally on both desktop GPUs and edge devices such as the **Jetson Orin Nano**.
It combines optional noise suppression, VAD, speaker diarization, transcription, anonymization, and meeting-report post-processing in a single, end-to-end workflow driven by Pyannote + Faster-Whisper.

---

## ✨ Features

* 🎧 Optional denoising with [RNNoise](https://github.com/xiph/rnnoise) and WAV normalization.
* 🔇 Voice activity detection (Silero/Marblenet) to trim silence before diarization.
* 👥 Speaker diarization powered by [Pyannote Audio](https://github.com/pyannote/pyannote-audio).
* 📝 Faster-Whisper transcription tuned for Jetson Orin and desktop GPUs.
* 🛡 Post-processing for anonymization, LLM-clean transcripts, and meeting reports.
* 🧱 Reproducible `uv` workspace with unit + integration tests.

---

## 📂 Repository Structure

```
ASR_jetson/
├── src/
│   └── asr_jetson/
│       ├── api/                 # FastAPI draft entrypoints
│       ├── asr/                 # Faster-Whisper model + decoding helpers
│       ├── config/              # Config dataclasses and prompt templates
│       ├── diarization/         # Pyannote diarization pipeline
│       ├── pipeline/            # End-to-end orchestration + CLI
│       ├── postprocessing/      # Anonymization, LLM clean-up, meeting reports
│       ├── preprocessing/       # Audio conversion + RNNoise interface
│       ├── utils/               # Shared helpers (logging, paths, metrics)
│       └── vad/                 # Silero / Marblenet voice activity detection
│
├── configs/                     # Runtime configuration samples
│   ├── dev.yaml
│   └── jetson.yaml
│
├── tests/                       # Pytest suite and fixtures
│   ├── data/                    # Sample audio + JSON fixtures
│   ├── test_full_pipeline.py
│   ├── test_meeting_report.py
│   ├── test_transformer_anonymizer.py
│   └── ...
│
├── scripts/                     # Utility scripts (export, profiling, tooling)
├── docker/                      # Dockerfiles for desktop + Jetson builds
├── models/                      # Local model cache (gitignored)
├── outputs/                     # Generated transcripts / reports (gitignored)
├── pyproject.toml               # Project metadata, extras, scripts
├── uv.lock                      # uv dependency lock file
└── README.md
```

---

## 🛠 Installation (with uv)

### Prerequisites

* Python ≥ 3.11
* CUDA-enabled GPU (recommended for realtime / large models)
* [ffmpeg](https://ffmpeg.org/) in PATH (required for audio conversion + RNNoise)
* [uv](https://github.com/astral-sh/uv) installed (`pip install uv`)
* `HUGGINGFACE_TOKEN` exported for private Pyannote pipelines
* Optional: `MISTRAL_API_KEY` when generating meeting reports via Mistral

### Setup

```bash
git clone https://github.com/XavBertrand/ASR_jetson.git
cd ASR_jetson

# Install dependencies (pick the variant matching your hardware)
uv sync --extra dev --extra media                                 # CPU-only
# uv sync --extra dev --extra media --extra gpu-linux             # Desktop GPU (CUDA)
# uv sync --extra dev --extra media --extra gpu-jetson            # Jetson Orin / aarch64

# Authenticate with Hugging Face once for Pyannote access
export HUGGINGFACE_TOKEN=hf_xxx

# Optional: enable meeting reports backed by Mistral
export MISTRAL_API_KEY=xxxx
```

---

## ▶️ Usage

### Run from CLI

```bash
uv run asr-pipeline \
  --audio path/to/file.wav \
  --out-dir outputs \
  --device cuda \
  --denoise \
  --speakers 2 \
  --whisper-model h2oai/faster-whisper-large-v3-turbo \
  --pyannote-pipeline pyannote/speaker-diarization-3.1 \
  --pyannote-token "$HUGGINGFACE_TOKEN"
```

* Switch `--device cpu` when CUDA is unavailable.
* Meeting reports require anonymization (enabled by default) and `MISTRAL_API_KEY`.
* For debugging, run `uv run python -m asr_jetson.pipeline.cli ...`.
* Jetson builds default to `pyannote/speaker-diarization-3.1` (Pyannote Audio 3.x) because `torchcodec` wheels are unavailable on aarch64; desktop x86_64 users can switch to `pyannote/speaker-diarization-community-1` (Pyannote 4.x) for the latest pipeline.

### Example Output

```json
{
  "json": "outputs/json/sample_pyannote_..._turbo.json",
  "srt": "outputs/srt/sample_pyannote_..._turbo.srt",
  "txt": "outputs/txt/sample_pyannote_..._turbo.txt",
  "txt_llm": "outputs/txt/sample_pyannote_..._turbo_clean.txt",
  "txt_anon": "outputs/txt/sample_pyannote_..._turbo_anon.txt",
  "txt_anon_llm": "outputs/txt/sample_pyannote_..._turbo_anon_clean.txt",
  "anon_mapping": "outputs/json/sample_pyannote_..._turbo_anon_mapping.json",
  "report_docx": "outputs/reports/sample_meeting_report.docx",
  "report_markdown": "outputs/reports/sample_meeting_report.md",
  "report_pdf": "outputs/pdf/sample_meeting_report.pdf",
  "report_status": "generated",
  "report_reason": ""
}
```

Running the pipeline writes diarized segments, transcripts, anonymized variants, and report artifacts under `outputs/`.  
If the Mistral prerequisites are missing, `report_status` becomes `skipped` and `report_reason` explains why (e.g., missing API key or unavailable endpoint).

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

* [ ] Enable low-latency / streaming inference for long recordings.
* [ ] Promote the FastAPI service into a deployable microservice.
* [ ] Extend anonymization to handle additional languages and entity types.
* [ ] Validate TensorRT / INT8 pipelines on Jetson for faster inference.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---
## 🙏 Acknowledgments

* [RNNoise](https://github.com/xiph/rnnoise) for lightweight denoising.
* [Pyannote Audio](https://github.com/pyannote/pyannote-audio) for speaker diarization.
* [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) / CTranslate2 for fast ASR.
* [Silero VAD](https://github.com/snakers4/silero-vad) and [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for VAD models.
* [uv](https://github.com/astral-sh/uv) for dependency management.
* [Mistral AI](https://github.com/mistralai/mistral-client) for meeting report generation.
