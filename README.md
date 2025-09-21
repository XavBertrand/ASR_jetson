# Lightweight ASR Pipeline with Diarization

This repository provides a **lightweight, modular, and efficient Automatic Speech Recognition (ASR) pipeline** designed to run locally on both desktop GPUs and edge devices such as the **Jetson Orin Nano**.  
It combines noise suppression, Voice Activity Detection (VAD), speaker diarization, and ASR transcription in a single, end-to-end workflow.

---

## ✨ Features

- 🎧 **Noise suppression** with [RNNoise](https://github.com/xiph/rnnoise) to enhance speech quality.  
- 🎙 **Voice Activity Detection (VAD)** using [Silero VAD](https://github.com/snakers4/silero-vad) for accurate speech segmentation.  
- 👥 **Speaker diarization** with **TitaNet-S embeddings** and **spectral clustering**, enabling speaker-attributed transcriptions.  
- 📝 **Automatic Speech Recognition (ASR)** using [FasterWhisper](https://github.com/SYSTRAN/faster-whisper) or **NVIDIA FastConformer** (via [NeMo](https://github.com/NVIDIA/NeMo)) for fast and accurate transcription.  
- 📦 **Lightweight & modular**: optimized for local use, Jetson deployment, or integration into existing apps.  
- 🧪 **Tested** with unit tests and integration tests (pytest).  

---

## 🛠 Pipeline Overview

1. **Noise Suppression**  
   Input audio is denoised with RNNoise to reduce background noise.  

2. **Voice Activity Detection (VAD)**  
   Silero VAD splits audio into speech / non-speech regions.  

3. **Speaker Embedding & Clustering**  
   - Each speech segment is processed with **TitaNet-S** to extract speaker embeddings.  
   - Segments are grouped using **spectral clustering** → speaker diarization.  

4. **ASR Transcription**  
   - Speech segments are transcribed using **FasterWhisper** (Whisper accelerated with CTranslate2) or **NeMo FastConformer**.  
   - Output includes **timestamps, text, and speaker labels**.  

---

## 📂 Repository Structure

```
├── scripts/
│   └──run_asr_pipeline.py # Wrapper for full pipeline execution
│
├── docker/
│   ├── Dockerfile           # Multi arch (x86_64 and arm64) docker file
│   └── requirements.txt     # python packages requirements
│
├── src/
│   ├── preprocessing/      # RNNoise wrapper
│   ├── vad/                # Silero VAD integration
│   ├── diarization/        # TitaNet-S embeddings + clustering
│   ├── asr/                # FasterWhisper / FastConformer ASR
│   ├── postprocessing/     # Text export functions
│   ├── pipeline/           # End-to-end pipeline orchestration
│   └── utils/              # Helper functions
│
├── tests/                  # Unit & integration tests (pytest)
│   ├── test_full_pipeline.py
│   ├── ...
│   └── data/               # Test audio files
│
├── models/                 # Some of the light AI models
│   ├── nemo/               # TitaNet-S weights
│   ├── rnnoise/            # RNNoise weigths
│
├── requirements.txt        # Dependencies
├── README.md               # Project documentation

```

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (for faster inference, optional but recommended)
- [ffmpeg](https://ffmpeg.org/) installed and in PATH

### Setup
```bash
git clone https://github.com/yourusername/asr-pipeline.git
cd asr-pipeline
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
.venv\Scripts\activate     # (Windows)
pip install -r requirements.txt
```

## Build multi-arch Docker image

Linux / macOS / WSL:
```bash
./docker/build.sh
```
Windows Powershell:
```bash
.\docker\build.ps1
```

## ▶️ Usage

### Run from CLI
```bash
python -m src.pipeline --audio_file path/to/file.wav --output transcript.json
```

### Run with Streamlit UI
```bash
streamlit run streamlit_app.py
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

## ✅ Testing

Integration tests ensure the pipeline works end-to-end.  

Run all tests:
```bash
pytest tests
```

---

## 📊 Benchmarks

| Model              | Device              | 1h audio runtime |
|--------------------|---------------------|------------------|
| Whisper Large      | Desktop GPU (4070) | ~12 min          |
| FasterWhisper-M    | Jetson Orin Nano    | ~25–30 min       |
| FastConformer-CTC  | Jetson Orin Nano    | ~20–25 min       |

*(Values are indicative and depend on audio quality & hardware setup)*

---

## 📌 Roadmap

- [ ] Add support for **online streaming transcription**  
- [ ] Extend diarization with **overlapping speech detection**  
- [ ] Add **speaker adaptation** (personalized profiles)  
- [ ] Docker container for **easy Jetson deployment**  

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Silero VAD](https://github.com/snakers4/silero-vad)  
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)  
- [TitaNet](https://arxiv.org/abs/2110.04410)  
- [Whisper & FasterWhisper](https://github.com/openai/whisper)  
- [RNNoise](https://github.com/xiph/rnnoise)  
