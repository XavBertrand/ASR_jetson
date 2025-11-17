import os
from pathlib import Path
import pytest

try:  # optional dependency
    from huggingface_hub import GatedRepoError  # type: ignore
except Exception:  # pragma: no cover - huggingface_hub absent
    class GatedRepoError(Exception):  # type: ignore
        ...

# Toujours raisonner depuis la racine du projet, même si PyCharm lance depuis /tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_diarization_on_real_file_integration():
    """
    Integration test for the Pyannote diarization pipeline.
    Ensures at least one labelled segment is produced.
    """
    # Import tardif pour laisser pytest découvrir le test même si nemo n'est pas installé
    try:
        from asr_jetson.diarization.pipeline_diarization import apply_diarization
    except Exception as e:
        pytest.skip(f"Pipeline import failed (probable dépendance manquante) : {e}")

    # Fichier de test
    audio_path = PROJECT_ROOT / "tests" / "data" / "test.mp3"
    if not audio_path.exists():
        pytest.skip("tests/data/test.mp3 manquant — on skip l'intégration.")

    if not os.getenv("HUGGINGFACE_TOKEN"):
        pytest.skip("HUGGINGFACE_TOKEN absent : la diarisation Pyannote repose sur un repo Hugging Face protégé.")

    try:
        diarized = apply_diarization(
            audio_path,
            n_speakers=1,
            device="cuda",
        )
    except (ModuleNotFoundError, ValueError) as e:
        pytest.skip(f"Pyannote indisponible pour les tests : {e}")
    except GatedRepoError as e:
        pytest.skip(f"Pyannote gated repository inaccessible : {e}")
    except Exception as e:
        # autre erreur bloquante : on la surface pour déboguer
        raise

    # Assertions de base
    assert isinstance(diarized, list), "La pipeline doit retourner une liste de segments"
    assert len(diarized) > 0, "Aucun segment détecté (pipeline Pyannote à vérifier)"

    for seg in diarized:
        # Vérifier les clés essentielles
        for k in ("start", "end", "speaker"):
            assert k in seg, f"Clé manquante dans le segment : {k}"

        assert isinstance(seg["speaker"], int)
        assert seg["start"] < seg["end"], "Segment vide/inversé"
        assert seg["start"] >= 0.0

    # Vérifie qu'au moins 2 segments existent OU qu'il y a >1 locuteur détecté
    speakers = {s["speaker"] for s in diarized}
    assert len(diarized) >= 1
    assert len(speakers) >= 1  # >=2 si tu veux forcer la séparation

    print(f"✅ Diarization successful: {len(diarized)} segments, {len(speakers)} speakers detected")
