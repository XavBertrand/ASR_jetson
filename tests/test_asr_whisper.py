# tests/test_asr_integration.py
import os
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.integration
def test_asr_on_real_file_and_attach_speakers():
    # imports tardifs pour éviter d’échouer à la découverte des tests
    try:
        from src.diarization.pipeline import apply_diarization
        from src.asr.whisper_engine import load_faster_whisper
        from src.asr.transcribe import transcribe_segments, attach_speakers
    except Exception as e:
        pytest.skip(f"Imports pipeline/ASR indisponibles : {e}")

    audio = PROJECT_ROOT / "tests" / "data" / "test.mp3"
    if not audio.exists():
        pytest.skip("tests/data/test.mp3 manquant")

    # 1) diarisation -> segments + speakers
    try:
        diar = apply_diarization(audio, n_speakers=2, device="cuda", clustering_method="spectral")
    except FileNotFoundError as e:
        pytest.skip(f"TitaNet indisponible : {e}")

    assert isinstance(diar, list) and len(diar) > 0

    # 2) ASR
    try:
        model, _meta = load_faster_whisper(model_name="tiny", device="cuda", compute_type="int8")
    except Exception as e:
        pytest.skip(f"faster-whisper indisponible : {e}")

    asr_segments = transcribe_segments(model, audio, diar, language=None)
    assert isinstance(asr_segments, list)
    # tolérant : il peut y avoir peu de texte selon le sample
    assert any(seg.get("text") for seg in asr_segments), "ASR n'a produit aucun texte"

    # 3) speaker attribution
    labeled = attach_speakers(diar, asr_segments)
    assert len(labeled) == len(asr_segments)
    # chaque entrée doit avoir un speaker
    assert all("speaker" in s for s in labeled)
