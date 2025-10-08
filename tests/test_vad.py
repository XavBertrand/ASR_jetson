import os
from src import load_silero_vad, apply_vad

from tests.conftest import PROJECT_ROOT


def test_vad_on_real_file():
    """
    Test Silero VAD sur un vrai fichier audio (tests/data/test.mp3).
    Vérifie qu'au moins un segment de voix est détecté.
    """
    input_file = os.path.join(PROJECT_ROOT, "tests", "data", "test.mp3")
    assert os.path.exists(input_file), "Le fichier de test test.mp3 est manquant dans tests/data/"

    # Charger modèle + VAD
    model, utils = load_silero_vad()
    sample_rate = 16000

    # Appliquer VAD
    segments = apply_vad(model, input_file, sample_rate)

    # Vérification
    assert isinstance(segments, list)
    assert len(segments) > 0, "Aucun segment vocal détecté dans test.mp3"
