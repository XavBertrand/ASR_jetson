"""Pyannote-based speaker diarization pipeline."""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


def _resolve_device(device: str) -> torch.device:
    """
    Resolve the requested device to an available :mod:`torch` device.

    :param device: Preferred device string such as ``"cpu"`` or ``"cuda"``.
    :type device: str
    :returns: Torch device respecting availability.
    :rtype: torch.device
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def apply_diarization(
    audio_path: str | Path,
    n_speakers: Optional[int] = None,
    device: str = "cuda",
    pyannote_pipeline: str = "pyannote/speaker-diarization-3.1",
    auth_token: Optional[str] = None,
) -> List[Dict[str, float | int]]:
    """
    Run the Pyannote diarization pipeline and return labelled segments.

    :param audio_path: Path to the audio file to analyse.
    :type audio_path: str | Path
    :param n_speakers: Optional constraint on the expected number of speakers.
    :type n_speakers: Optional[int]
    :param device: Execution device hint (``"cpu"`` or ``"cuda"``).
    :type device: str
    :param pyannote_pipeline: Name of the pretrained Pyannote pipeline to load.
    :type pyannote_pipeline: str
    :param auth_token: Hugging Face authentication token. Falls back to the
        ``HUGGINGFACE_TOKEN`` environment variable when ``None``.
    :type auth_token: Optional[str]
    :returns: List of diarized segments with ``start``/``end`` (seconds) and ``speaker`` id.
    :rtype: List[Dict[str, float | int]]
    """
    from pyannote.audio import Pipeline

    token = auth_token if auth_token is not None else HF_TOKEN

    print("=" * 40 + "\n" + "   PYANNOTE DIARIZATION\n" + "=" * 40)

    pipeline = Pipeline.from_pretrained(pyannote_pipeline, use_auth_token=token)
    pipeline.to(_resolve_device(device))

    inference_inputs = {"audio": str(audio_path)}
    if n_speakers and n_speakers > 0:
        annotation = pipeline(inference_inputs, num_speakers=n_speakers)
    else:
        annotation = pipeline(inference_inputs)

    label_map: Dict[str, int] = {}
    next_label = 0
    results: List[Dict[str, float]] = []

    for segment, _, label in annotation.itertracks(yield_label=True):
        label_str = str(label)
        if label_str.isdigit():
            speaker_id = int(label_str)
        else:
            if label_str not in label_map:
                label_map[label_str] = next_label
                next_label += 1
            speaker_id = label_map[label_str]

        results.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": speaker_id,
            }
        )

    results.sort(key=lambda item: (item["start"], item["end"]))

    torch.cuda.empty_cache()
    gc.collect()

    return results
