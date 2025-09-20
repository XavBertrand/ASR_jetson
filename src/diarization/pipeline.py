# src/diarization/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal

from src.preprocessing.vad import load_silero_vad, apply_vad
from src.diarization.titanet import load_titanet, extract_embeddings
from src.diarization.clustering import cluster_embeddings


def apply_diarization(
    audio_path: str | Path,
    n_speakers: int = 2,
    device: str = "cpu",
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

    # 2) Embeddings TitaNet-S
    spk_model = load_titanet(device=device)  # la fonction met le modèle en .eval()
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

    return diarized_segments
