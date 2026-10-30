"""Parakeet TDT v3 ONNX inference, independent of the Torch/Transformers stack.

CPU uses the INT8 export; CUDA uses floating-point weights and the CUDA runtime.
Weights are downloaded only when selected, never while building an image.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any

MODEL_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
CUDA_MODEL_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3"
CUDA_MODEL_FILES = ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt", "encoder.weights")


def _require_cuda_runtime(sherpa: Any) -> None:
    """Reject CPU-only wheels and missing CUDA libraries before native inference."""
    import torch

    if "+cuda" not in sherpa.__version__:
        raise RuntimeError("Parakeet CUDA requires: uv sync --extra parakeet-cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Parakeet requested CUDA but no CUDA GPU is available")
    # Inspect Sherpa's own runtime, not the unrelated Python onnxruntime used
    # by Whisper VAD. ORT must initialize its provider bridge itself.
    provider = Path(sherpa.__file__).parent / "lib" / "libonnxruntime_providers_cuda.so"
    if not provider.is_file():
        raise RuntimeError(f"Parakeet CUDA provider is missing: {provider}")


def _create_cuda_recognizer(factory: Any, **kwargs: Any) -> Any:
    """Detect Sherpa's native CPU fallback diagnostics during initialization.

    Sherpa reports a provider initialization failure on native stderr and can
    otherwise return a CPU recognizer. Preserve these diagnostics and fail the
    explicitly requested CUDA run. The CLI initializes its ASR model serially.
    """
    with TemporaryFile() as diagnostics:
        saved_stderr = os.dup(2)
        sys.stderr.flush()
        try:
            os.dup2(diagnostics.fileno(), 2)
            model = factory(**kwargs)
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
            diagnostics.seek(0)
            log = diagnostics.read().decode("utf-8", errors="replace")
            if log:
                sys.stderr.write(log)
                sys.stderr.flush()
    if "fallback to cpu" in log.lower():
        raise RuntimeError(
            "Parakeet CUDA initialization failed; refusing CPU fallback. "
            "Check the CUDA/cuDNN libraries and GPU runtime diagnostics above."
        )
    return model


def load_parakeet(model_dir: str | Path | None = None, device: str = "cpu") -> Any:
    """Load a device-appropriate export without silently changing the device."""
    if device not in {"cpu", "cuda"}:
        raise ValueError(
            "Parakeet device must be cpu or cuda; use CUDA_VISIBLE_DEVICES to select a GPU"
        )
    try:
        import sherpa_onnx
    except ImportError as exc:
        extra = "parakeet-cuda" if device == "cuda" else "parakeet"
        raise RuntimeError(f"Parakeet requires: uv sync --extra {extra}") from exc

    if device == "cuda":
        _require_cuda_runtime(sherpa_onnx)
    files = CUDA_MODEL_FILES if device == "cuda" else MODEL_FILES
    repo = CUDA_MODEL_REPO if device == "cuda" else MODEL_REPO

    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(repo, allow_patterns=list(files))
    root = Path(model_dir)
    for name in files:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing Parakeet model file: {root / name}")
    options = {
        "encoder": str(root / files[0]),
        "decoder": str(root / files[1]),
        "joiner": str(root / files[2]),
        "tokens": str(root / files[3]),
        "model_type": "nemo_transducer",
        "decoding_method": "greedy_search",
        "num_threads": 2,
        "provider": device,
    }
    factory = sherpa_onnx.OfflineRecognizer.from_transducer
    if device == "cuda":
        return _create_cuda_recognizer(factory, **options)
    return factory(**options)


def transcribe_parakeet(
    model: Any,
    wav_path: str | Path,
    diar_segments: list[dict[str, Any]],
    chunk_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    """Decode bounded speech windows, preserving absolute audio timestamps.

    Diarization supplies speech activity in place of Whisper's internal VAD.
    Overlapping turns are merged to avoid transcribing the same audio twice.
    Segment boundaries describe decoded audio windows, not forced word alignment.
    """
    import soundfile as sf

    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("parakeet_chunk_seconds must be finite and positive")
    results = []
    with sf.SoundFile(str(wav_path)) as audio:
        if audio.samplerate != 16000 or audio.channels != 1:
            raise ValueError("Parakeet requires 16 kHz mono audio")
        windows: list[list[int]] = []
        for seg in sorted(diar_segments, key=lambda s: float(s["start"])):
            start = max(0, min(len(audio), round(float(seg["start"]) * 16000)))
            end = max(0, min(len(audio), round(float(seg["end"]) * 16000)))
            if end <= start:
                continue
            if windows and start < windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], end)
            else:
                windows.append([start, end])
        chunk_samples = max(1, round(chunk_seconds * 16000))
        for start, end in windows:
            for offset in range(start, end, chunk_samples):
                count = min(chunk_samples, end - offset)
                audio.seek(offset)
                samples = audio.read(count, dtype="float32")
                stream = model.create_stream()
                stream.accept_waveform(16000, samples)
                model.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    results.append(
                        {
                            "start": offset / 16000.0,
                            "end": (offset + count) / 16000.0,
                            "text": text,
                        }
                    )
    return results
