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
# uv sync --extra dev --extra gpu-jetson                          # Jetson Orin / aarch64
# Note: l'extra "media" (PyAV) requiert FFmpeg 7 et n'est pas dispo sur Jetson.

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
  --whisper-model h2oai/faster-whisper-large-v3-turbo \
  --whisper-compute int8_float16 \
  --lang fr \
  --denoise \
  --speakers 2 \
  --pyannote-pipeline pyannote/speaker-diarization-3.1 \
  --pyannote-token "$HUGGINGFACE_TOKEN" \
  --asr-prompt "Keywords: Kleos, DGA, space." \
  --speaker-context "SPK_1 (sales lead) interviewing SPK_2 (candidate)" \
  --meeting-date 2024-05-10 \
  --meeting-report-type entretien_collaborateur \
  --monitor-gpu-memory
```

Argument reference (mirrors `asr_jetson.pipeline.cli` exactly):

- `--audio` (required): Input audio (wav/mp3/flac); converted to WAV automatically.
- `--out-dir` (default `outputs`): Root directory for JSON/SRT/TXT and reports.
- `--device` (default `cuda`): Uses CUDA when available, otherwise falls back to CPU. Use `--device cpu` to force CPU.
- `--whisper-model` (default `h2oai/faster-whisper-large-v3-turbo`): Faster-Whisper model id (e.g., `openai/whisper-large-v3-turbo`, `medium`).
- `--whisper-compute` (default `int8_float16`): CTranslate2 `compute_type`; on GPU `int8` is auto-sanitized to `int8_float16`, and an unsupported value falls back to `float16`. On CPU, `int8` is the practical choice.
- `--lang` (default `fr`): Forced transcription language (ISO code). The decoder always forces a language; set this to `en`, `es`, etc. as needed.
- `--denoise`: Enable RNNoise preprocessing before diarization/ASR.
- `--speakers`: Optional expected speaker count forwarded to Pyannote.
- `--pyannote-pipeline` (default `pyannote/speaker-diarization-3.1`): Pyannote pipeline id. `pyannote/speaker-diarization-community-1` (Pyannote 4.x) is available on x86_64.
- `--pyannote-token`: Hugging Face token; falls back to the `HUGGINGFACE_TOKEN` env var when omitted.
- `--asr-prompt`: Initial prompt passed to Faster-Whisper to bias decoding; defaults to `Kleos, Pennylane, CJD, Manupro, El Moussaoui`.
- `--speaker-context`: Optional speaker/role description injected (after anonymization) into the meeting report prompt.
- `--meeting-date`: Reference date (`YYYY-MM-DD`) used in the report prompt and filenames; defaults to today's date.
- `--meeting-report-type`: Meeting report prompt category (`entretien_collaborateur`, `entretien_client_particulier_contentieux`, `entretien_client_professionnel_conseil`, `entretien_client_professionnel_contentieux`); defaults to `entretien_collaborateur`.
- `--monitor-gpu-memory`: Print per-stage CUDA memory usage (only when CUDA is available).

Meeting report generation is enabled by default and requires both `MISTRAL_API_KEY` and the `mistralai` package; missing prerequisites cause a runtime error (there is no CLI switch to disable the report stage).
For debugging, run `uv run python -m asr_jetson.pipeline.cli ...`.

### Example Output

```json
{
  "json": "outputs/json/sample_pyannote_speaker-diarization-3.1_h2oai_faster-whisper-large-v3-turbo.json",
  "srt": "outputs/srt/sample_pyannote_speaker-diarization-3.1_h2oai_faster-whisper-large-v3-turbo.srt",
  "txt": "outputs/txt/sample_pyannote_speaker-diarization-3.1_h2oai_faster-whisper-large-v3-turbo.txt",
  "txt_llm": "outputs/txt/sample_pyannote_speaker-diarization-3.1_h2oai_faster-whisper-large-v3-turbo_clean.txt",
  "txt_anon": "outputs/txt/sample_pyannote_speaker-diarization-3.1_h2oai_faster-whisper-large-v3-turbo_anon.txt",
  "txt_anon_llm": "outputs/txt/sample_pyannote_speaker-diarization-3.1_h2oai_faster-whisper-large-v3-turbo_anon_clean.txt",
  "anon_mapping": "outputs/json/sample_pyannote_speaker-diarization-3.1_h2oai_faster-whisper-large-v3-turbo_anon_mapping.json",
  "report_anonymized_txt": "outputs/reports/sample_pyannote_speaker-diarization-3.1_h2oai_faster-whisper-large-v3-turbo_meeting_report_anonymized.md",
  "report_markdown": "outputs/reports/sample_pyannote_speaker-diarization-3.1_h2oai_faster-whisper-large-v3-turbo_meeting_report.md",
  "report_pdf": "outputs/pdf/compte_rendu_sample_2024-05-10_154233.pdf",
  "report_docx": "outputs/pdf/compte_rendu_sample_2024-05-10_154233.docx",
  "report_txt": null,
  "report_status": "generated",
  "report_reason": ""
}
```

Running the pipeline writes diarized segments, transcripts, anonymized variants, and report artifacts under `outputs/`.
Report filenames embed the audio stem, meeting date, and timestamp (e.g., `compte_rendu_<audio>_<date>_<time>.pdf`/`.docx`).

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
