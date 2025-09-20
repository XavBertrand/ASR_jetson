from typing import List, Dict
from src.preprocessing.vad import load_silero_vad, apply_vad
from src.diarization.titanet import load_titanet, extract_embeddings
from src.diarization.clustering import cluster_speakers

from tests.conftest import PROJECT_ROOT


def apply_diarization(wav_path: str, n_speakers: int = 2, device: str = "cpu") -> List[Dict]:
    """
    Full diarization pipeline:
    - Run VAD
    - Extract embeddings with TitaNet-S
    - Cluster speakers
    - Return diarized segments
    """
    model, _ = load_silero_vad()
    vad_segments = apply_vad(model, wav_path, sample_rate=16000)

    titanet = load_titanet(device)
    embeddings = extract_embeddings(titanet, wav_path, vad_segments, device)

    labels = cluster_speakers(embeddings, n_speakers)

    diarized = []
    for seg, label in zip(vad_segments, labels):
        diarized.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": f"SPEAKER_{label}"
        })

    return diarized
