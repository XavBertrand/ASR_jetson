# src/diarization/pipeline_diarization.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal
import soundfile as sf

from src.asr_jetson.vad.silero import load_silero_vad, apply_vad
from src.asr_jetson.diarization.titanet import load_titanet, extract_embeddings
from src.asr_jetson.diarization.clustering import cluster_embeddings


def _ensure_seconds(diar_segments, wav_path, sr_hint=16000):
    """
    Convertit les segments de diarization en SECONDES si on détecte qu'ils sont en échantillons.
    Heuristique : si les timestamps ressemblent à des indices d'échantillons (>> secondes), on divise par sr.
    """
    if not diar_segments:
        return diar_segments

    # essaie de lire le sample rate réel du fichier
    sr = sr_hint
    try:
        with sf.SoundFile(str(wav_path)) as f:
            sr = int(f.samplerate) or sr_hint
    except Exception:
        pass

    # si les timestamps semblent être en samples (par ex. très grands)
    max_end = max(float(d.get("end", 0)) for d in diar_segments)
    # seuil simple : si > 10 * sr, on considère que c'est des samples (>=10s en samples)
    if max_end > 10 * sr:
        for d in diar_segments:
            d["start"] = float(d["start"]) / sr
            d["end"] = float(d["end"]) / sr
    return diar_segments


def apply_diarization(
    audio_path: str | Path,
    n_speakers: int = 2,
    device: str = "cuda",
    clustering_method: Literal["spectral", "kmeans"] = "spectral",
) -> List[Dict]:
    """
    Pipeline: VAD -> embeddings TitaNet-S -> clustering -> segments étiquetés.

    Retourne une liste de dicts :
      { 'start': int, 'end': int, 'speaker': int }
    Les temps sont en échantillons à 16 kHz (cohérents avec la VAD et TitaNet).
    """
    # 1) VAD sur un mono/16 kHz (apply_vad gère la lecture et resample si besoin)
    audio_path = str(Path(audio_path))
    vad_model, _ = load_silero_vad()
    vad_segments = apply_vad(vad_model, audio_path, sample_rate=16000)

    if not vad_segments:
        return []

    # 2) Embeddings TitaNet-S (cherche d’abord localement)
    local_dir = os.getenv("ASR_NEMO_DIR", str(Path(__file__).resolve().parents[3] / "models" / "nemo"))
    spk_model = load_titanet(device=device, local_dir=local_dir)  # .eval() déjà appliqué
    embeddings = extract_embeddings(spk_model, audio_path, vad_segments, device=device)

    if embeddings is None or len(embeddings) == 0:
        return []

    # 3) Clustering (propage la méthode demandée)
    labels = cluster_embeddings(embeddings, n_speakers=n_speakers, method=clustering_method)

    # 4) Assemblage des segments
    diarized_segments: List[Dict] = []
    for seg, label in zip(vad_segments, labels):
        diarized_segments.append(
            {"start": int(seg["start"]), "end": int(seg["end"]), "speaker": int(label)}
        )

    diarized_segments = _ensure_seconds(diarized_segments, audio_path)

    return diarized_segments
