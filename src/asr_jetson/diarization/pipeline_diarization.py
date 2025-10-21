"""Pyannote-based speaker diarization pipeline."""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, List, Optional
from tempfile import TemporaryDirectory

import torch

from asr_jetson.preprocessing.convert_to_wav import convert_to_wav

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
    # pyannote_pipeline: str = "pyannote/speaker-diarization-3.1",
    pyannote_pipeline: str = "pyannote/speaker-diarization-community-1",
    auth_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN"),
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

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    temp_dir: Optional[TemporaryDirectory] = None
    processed_audio_path = audio_path

    try:
        if audio_path.suffix.lower() not in {".wav", ".wave"}:
            temp_dir = TemporaryDirectory()
            temp_wav_path = Path(temp_dir.name) / f"{audio_path.stem}.wav"
            processed_audio_path = convert_to_wav(audio_path, temp_wav_path)
        else:
            processed_audio_path = audio_path

        token = auth_token if auth_token is not None else HF_TOKEN

        print("=" * 40 + "\n" + "   PYANNOTE DIARIZATION\n" + "=" * 40)

        pipeline = Pipeline.from_pretrained(pyannote_pipeline, token=token)
        pipeline.to(_resolve_device(device))

        inference_inputs = {"audio": str(processed_audio_path)}
        if n_speakers and n_speakers > 0:
            annotation = pipeline(inference_inputs, num_speakers=n_speakers)
        else:
            annotation = pipeline(inference_inputs)

        label_map: Dict[str, int] = {}
        next_label = 0
        results: List[Dict[str, float]] = []

        def _append_interval(start_s: float, end_s: float, label_obj) -> None:
            nonlocal next_label
            label_str = str(label_obj)
            if label_str not in label_map:
                label_map[label_str] = next_label
                next_label += 1
            speaker_id = label_map[label_str]
            results.append({"start": float(start_s), "end": float(end_s), "speaker": int(speaker_id)})

        # Case 1: "legacy" (pyannote 3.x): Annotation with .itertracks(...)
        if hasattr(annotation, "itertracks"):
            for segment, _, label in annotation.itertracks(yield_label=True):
                _append_interval(segment.start, segment.end, label)
        else:
            # Cas 2: pyannote >= 4 (community-1): DiarizeOutput with .speaker_diarization
            diar_obj = getattr(annotation, "speaker_diarization", None)
            if diar_obj is None and isinstance(annotation, dict):
                diar_obj = annotation.get("speaker_diarization", None)
            if diar_obj is None:
                raise TypeError(
                    "Unexpected diarization output. Expected Annotation.itertracks or DiarizeOutput.speaker_diarization."
                )
            for turn, speaker in diar_obj:
                _append_interval(turn.start, turn.end, speaker)

        results.sort(key=lambda item: (item["start"], item["end"]))

        torch.cuda.empty_cache()
        gc.collect()

        return results
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
