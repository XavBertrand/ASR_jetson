# src/preprocessing/silero.py

import torch
from pathlib import Path
from typing import List, Dict, Optional, Any


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

def merge_segments(
    segments: List[Dict],
    *,
    merge_gap_s: float = 0.25,       # si l’écart entre deux segments ≤ ce seuil → fusion
    min_dur_s: float = 0.10,         # on jette les segments < min_dur_s après fusion
    pad_s: float = 0.0,              # padding facultatif (appliqué puis recoupé)
    require_same_speaker: bool = True,  # fusionne seulement si speaker identique
) -> List[Dict]:
    """
    Fusionne des segments temporels (en secondes) trop proches / qui se chevauchent.
    - merge_gap_s : écart max pour fusionner deux segments adjacents
    - min_dur_s   : durée minimale à conserver après fusion
    - pad_s       : pad (±) appliqué autour de chaque segment avant fusion
    - require_same_speaker : si True, on fusionne seulement si seg['speaker'] est identique
    """
    if not segments:
        return []

    # normalise + trie
    norm = []
    for s in segments:
        st, en = float(s["start"]), float(s["end"])
        if en <= st:  # ignore invalid
            continue
        spk = s.get("speaker")
        # padding local
        st -= pad_s
        en += pad_s
        norm.append({"start": st, "end": en, "speaker": spk, **{k: v for k, v in s.items() if k not in ("start", "end", "speaker")}})
    norm.sort(key=lambda x: (x.get("speaker"), x["start"])) if require_same_speaker else norm.sort(key=lambda x: x["start"])

    merged: List[Dict] = []
    cur = norm[0]

    def _can_merge(a: Dict, b: Dict) -> bool:
        if require_same_speaker and a.get("speaker") != b.get("speaker"):
            return False
        gap = b["start"] - cur["end"]
        return gap <= merge_gap_s  # fusionne si recouvrement (gap<0) ou petit trou

    for nxt in norm[1:]:
        if _can_merge(cur, nxt):
            # étend la fenêtre courante
            cur["end"] = max(cur["end"], nxt["end"])
            # si tu veux concaténer un éventuel texte : cur["text"] = (cur.get("text","") + " " + nxt.get("text","")).strip()
        else:
            # enlève le pad avant d’ajouter
            start, end = max(0.0, cur["start"] + 0.0), cur["end"] - 0.0
            if (end - start) >= min_dur_s:
                cur["start"], cur["end"] = start, end
                merged.append(cur)
            cur = nxt

    # flush dernier
    start, end = max(0.0, cur["start"] + 0.0), cur["end"] - 0.0
    if (end - start) >= min_dur_s:
        cur["start"], cur["end"] = start, end
        merged.append(cur)

    # retire le padding si utilisé
    if pad_s != 0.0:
        for s in merged:
            s["start"] = max(0.0, s["start"] + 0.0 - pad_s)
            s["end"] = s["end"] - 0.0 + pad_s

    # tri final par temps
    merged.sort(key=lambda x: x["start"])
    return merged


def apply_vad(
    model: torch.jit.ScriptModule,
    wav_path: Path,
    sample_rate: int = 16000,
    *,
    threshold: float = 0.5,
    min_silence_duration_ms: int = 150,  # ↑ pour MERGER davantage (ex: 300–600)
    speech_pad_ms: int = 40,             # ↓ pour réduire l’OVERLAP (ex: 10–30)
    window_size_samples: int = 512,      # défaut Silero (10–40 ms selon sr)
    return_seconds: bool = True,         # pratique pour aligner avec Whisper
) -> List[Dict[str, Any]]:
    """
    Applique Silero VAD avec contrôle fin de l'overlap et du merge.
    """
    _, utils = load_silero_vad()
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(str(wav_path), sampling_rate=sample_rate)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        window_size_samples=window_size_samples,
        return_seconds=return_seconds,  # => start/end en secondes
    )

    return speech_timestamps
