import json
from pathlib import Path
import pytest

# Racine du projet même si le test est lancé depuis /tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.integration
def test_full_pipeline_end_to_end(tmp_path):
    """
    Test d'intégration end-to-end :
    denoise (off) -> diarization (VAD + TitaNet + clustering) -> ASR (faster-whisper)
    -> fusion -> export JSON & SRT.

    Le test skip proprement si une dépendance lourde (NeMo, modèles) n'est pas dispo.
    """
    # Import tardif pour éviter un ImportError à la collection
    try:
        from src.asr_jetson.pipeline.full_pipeline import PipelineConfig, run_pipeline
    except Exception as e:
        pytest.skip(f"Import pipeline impossible (dépendances manquantes ?) : {e}")

    # Fichier audio de test
    audio_path = PROJECT_ROOT / "tests" / "data" / "test.mp3"
    if not audio_path.exists():
        pytest.skip("tests/data/test.mp3 manquant — on skip l'intégration.")

    # Config "safe" pour CI : CPU, denoise désactivé (évite soucis ffmpeg/arnndn)
    out_dir = tmp_path / "outputs"
    cfg = PipelineConfig(
        denoise=False,
        device="cuda",
        n_speakers=1,
        clustering_method="spectral",
        spectral_assign_labels="kmeans",
        vad_min_chunk_s=0.5,
        whisper_model="medium",
        whisper_compute="int8_float16",  # <--- clé : compatible CUDA
        language=None,
        out_dir=out_dir,
    )

    # Run
    try:
        result = run_pipeline(str(audio_path), cfg)
    except FileNotFoundError as e:
        # typiquement : modèle NeMo TitaNet indisponible/non téléchargeable dans l'environnement de test
        pytest.skip(f"Ressource modèle indisponible pour l'intégration : {e}")
    except Exception:
        # On laisse les autres exceptions remonter pour debug
        raise

    # Assertions de base sur la structure
    assert isinstance(result, dict)
    for k in ("diarization", "asr", "labeled"):
        assert k in result, f"Clé manquante dans le résultat : {k}"
        assert isinstance(result[k], list), f"{k} doit être une liste"

    # On s'attend à au moins 1 segment final taggé
    assert len(result["labeled"]) > 0, "Aucun segment final — vérifier VAD/embeddings/ASR/fusion"

    # Vérification SRT/JSON
    assert "json" in result and "srt" in result
    json_path = Path(result["json"])
    srt_path = Path(result["srt"])
    assert json_path.exists(), "Le JSON d'export n'existe pas"
    assert srt_path.exists(), "Le SRT d'export n'existe pas"

    # Contenu JSON cohérent
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "segments" in payload and isinstance(payload["segments"], list)
    assert len(payload["segments"]) == len(result["labeled"])
    # Sanity sur le premier segment
    first = payload["segments"][0]
    for key in ("start", "end", "start_s", "end_s", "speaker"):
        assert key in first, f"Clé manquante dans un segment JSON : {key}"

    # Contenu SRT : timestamps et speaker visibles
    srt_txt = srt_path.read_text(encoding="utf-8")
    assert "-->" in srt_txt, "Format SRT invalide (timestamps manquants)"
    assert "SPK" in srt_txt, "Les identifiants locuteurs ne sont pas présents dans le SRT"

    # Idempotence simple : relance avec les mêmes paramètres ne doit pas lever d'exception
    result2 = run_pipeline(str(audio_path), cfg)
    assert isinstance(result2, dict)
