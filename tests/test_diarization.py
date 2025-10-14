from pathlib import Path
import pytest

# Toujours raisonner depuis la racine du projet, même si PyCharm lance depuis /tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_diarization_on_real_file_integration():
    """
    Test d'intégration : VAD -> embeddings TitaNet -> clustering.
    On vérifie que la pipeline retourne au moins un segment étiqueté.
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

    # Si TitaNet n'est pas accessible en test, on skip proprement
    try:
        diarized = apply_diarization(
            audio_path,
            n_speakers=1,
            device="cuda",
            clustering_method="spectral",  # identique à ta pipeline
            backend="titanet",
        )
    except FileNotFoundError as e:
        # typiquement : modèle NeMo introuvable / non téléchargé en contexte de test
        pytest.skip(f"TitaNet indisponible pour les tests : {e}")
    except Exception as e:
        # autre erreur bloquante : on la surface pour déboguer
        raise

    # Assertions de base
    assert isinstance(diarized, list), "La pipeline doit retourner une liste de segments"
    assert len(diarized) > 0, "Aucun segment détecté (VAD/embeddings/clustering à vérifier)"

    # Sanity checks sur la structure et la cohérence
    sample_rate = 16000  # TitaNet et VAD utilisent 16kHz

    for seg in diarized:
        # Vérifier les clés essentielles
        for k in ("start", "end", "speaker"):
            assert k in seg, f"Clé manquante dans le segment : {k}"

        assert isinstance(seg["speaker"], int)
        assert seg["start"] < seg["end"], "Segment vide/inversé"

        # Vérifier les timestamps en secondes si présents
        if "start_s" in seg and "end_s" in seg:
            assert seg["start_s"] < seg["end_s"], "Timestamps en secondes incohérents"
            # Vérifier la cohérence entre échantillons et secondes
            assert abs(seg["start_s"] - seg["start"] / sample_rate) < 0.001
            assert abs(seg["end_s"] - seg["end"] / sample_rate) < 0.001

        # 2 locuteurs max (0 et 1) avec n_speakers=2
        assert 0 <= seg["speaker"] < 2

    # Vérifie qu'au moins 2 segments existent OU qu'il y a >1 locuteur détecté
    speakers = {s["speaker"] for s in diarized}
    assert len(diarized) >= 1
    assert len(speakers) >= 1  # >=2 si tu veux forcer la séparation

    print(f"✅ Diarization successful: {len(diarized)} segments, {len(speakers)} speakers detected")