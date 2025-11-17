# src/preprocessing/silero.py

import torch
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple


def load_silero_vad() -> Tuple[torch.jit.ScriptModule, tuple]:
    """
    Load the Silero VAD model and utilities from torch.hub.

    Returns
    -------
    model : torch.jit.ScriptModule
        The pre-trained Silero VAD model.
    utils : tuple
        Helper functions (get_speech_timestamps, save_audio_chunks, read_audio, VADIterator, collect_chunks).
    """
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model, utils


def normalize_segments(
    segments: List[Dict[str, float]],
    *,
    merge_gap_ms: int = 140,
    min_speech_ms: int = 140,
    pad_ms: int = 100,
    total_sec: float | None = None,
) -> List[Dict[str, float]]:
    """Post-traitement harmonisé pour mieux capter les alternances:
    - padding de début/fin
    - fusion des gaps courts
    - filtre par durée minimale
    """
    if not segments:
        return []
    segs = sorted(segments, key=lambda d: d["start"])

    # Padding
    pad = pad_ms / 1000.0
    for s in segs:
        s["start"] = max(0.0, s["start"] - pad)
        s["end"] = s["end"] + pad
        if total_sec is not None:
            s["end"] = min(total_sec, s["end"])

    # Fusion des gaps courts
    merged = [segs[0]]
    max_gap = merge_gap_ms / 1000.0
    for s in segs[1:]:
        if s["start"] - merged[-1]["end"] <= max_gap:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s)

    # Filtre durée mini
    min_len = min_speech_ms / 1000.0
    merged = [s for s in merged if (s["end"] - s["start"]) >= min_len]
    return merged


@torch.no_grad()
def apply_vad(
    model: torch.jit.ScriptModule,
    wav_path: str | Path,
    sample_rate: int = 16000,
    *,
    # --- Paramètres natifs Silero ---
    threshold: float = 0.5,
    min_silence_duration_ms: int = 150,
    speech_pad_ms: int = 40,
    window_size_samples: int = 512,
    return_seconds: bool = True,
    # --- Pour éviter de recharger via hub à chaque appel ---
    utils: tuple | None = None,
    # --- Post-traitement harmonisé (preset alternances) ---
    postprocess: bool = True,
    merge_gap_ms: int = 140,
    min_speech_ms: int = 140,
    pad_ms: int = 100,
) -> List[Dict[str, Any]]:
    """
    Applique Silero VAD puis un post-traitement harmonisé (padding + fusion + min durée)
    pour obtenir plus d'alternances propres entre locuteurs.

    Retourne une liste de segments: [{'start': sec, 'end': sec}, ...]
    """
    # Récupération des utils Silero (sans recharger le modèle si déjà fourni)
    if utils is None:
        _model_unused, utils = load_silero_vad()
    get_speech_timestamps, _, read_audio, _, _ = utils

    # Lecture audio
    wav_path = str(wav_path)
    wav = read_audio(wav_path, sampling_rate=sample_rate)
    total_sec = len(wav) / float(sample_rate)

    # Inference Silero
    raw_segments = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        window_size_samples=window_size_samples,
        return_seconds=return_seconds,  # => start/end en secondes
    )

    if not postprocess:
        return raw_segments

    # Post-traitement preset (merge/pad/min_length)
    final_segments = normalize_segments(
        raw_segments,
        merge_gap_ms=merge_gap_ms,
        min_speech_ms=min_speech_ms,
        pad_ms=pad_ms,
        total_sec=total_sec,
    )
    return final_segments
