# src/preprocessing/vad.py

import torch
from pathlib import Path
from typing import List, Dict, Any


def load_silero_vad():
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


def apply_vad(
    model: torch.jit.ScriptModule,
    wav_path: Path,
    sample_rate: int = 16000,
) -> List[Dict[str, Any]]:
    """
    Apply Silero VAD to detect speech segments in an audio file.

    Parameters
    ----------
    model : torch.jit.ScriptModule
        Silero VAD model.
    wav_path : Path
        Path to the input WAV/OGG/MP3 file.
    sample_rate : int, optional
        Target sample rate (default: 16000).

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries with start/end samples of detected speech.
    """
    # Récupérer les utils
    _, utils = load_silero_vad()
    (get_speech_timestamps, _, read_audio, _, _) = utils

    # Lire l’audio
    wav = read_audio(str(wav_path), sampling_rate=sample_rate)

    # Appliquer la détection VAD
    speech_timestamps = get_speech_timestamps(
        wav, model, sampling_rate=sample_rate
    )

    return speech_timestamps
