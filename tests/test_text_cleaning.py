import io
from pathlib import Path
import shutil
import os
import sys
import json
import pytest

# Si ton projet n'ajoute pas automatiquement la racine au PYTHONPATH :
# On pousse la racine (dossier contenant text_export.py) dans sys.path pour l'import.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocessing.llm_clean import clean_transcript_with_llm  # <-- adapte si nécessaire


class FakeLLM:
    """LLM factice, totalement déterministe, pour tests unitaires."""
    def __init__(self, mapping=None):
        # mapping permet de transformer certaines entrées en sorties connues
        self.mapping = mapping or {}

    def __call__(self, prompt: str) -> str:
        # Renvoie un texte "corrigé" très simple, pour rendre le test robuste
        base = self.mapping.get(prompt, prompt)
        # Ex. minuscule correction de ponctuation/espaces doubles:
        corrected = (
            base.replace(" ,", ",")
                .replace("  ", " ")
                .replace("d accord", "d’accord")
                .replace("OK", "Ok")
                .strip()
        )
        # Ajoute un point final si manquant et que ça ressemble à une phrase
        if corrected and corrected[-1].isalnum():
            corrected += "."
        return corrected


@pytest.fixture
def sample_text():
    return (
        "Bonjour  , ceci est un   test de transcription d accord\n"
        "OK ça marche encore ?\n"
        "fin de test"
    )


def test_writes_new_file_and_keeps_original(tmp_path: Path, sample_text: str):
    # 1) Prépare un fichier source
    src = tmp_path / "export.txt"
    src.write_text(sample_text, encoding="utf-8")

    # 2) Fake LLM
    llm = FakeLLM()

    # 3) Appelle la fonction sans output_path => doit écrire test_clean.txt à côté
    out_path_str = clean_transcript_with_llm(input_path=src, output_path=None, llm=llm)
    out_path = Path(out_path_str)

    # 4) Vérifications
    assert src.exists(), "Le fichier source ne doit pas être écrasé."
    assert out_path.name == "test_clean.txt", "Le nom de fichier de sortie par défaut doit être test_clean.txt."
    assert out_path.exists(), "Le fichier de sortie doit être créé."

    # 5) Contenu : doit être 'corrigé' par le FakeLLM (déterministe)
    cleaned = out_path.read_text(encoding="utf-8")
    assert "d’accord" in cleaned
    assert "Bonjour, ceci est un test de transcription d’accord." in cleaned.replace("\n", " ")
    assert cleaned.endswith("."), "On attend un point final ajouté par le LLM factice."


def test_custom_output_path(tmp_path: Path, sample_text: str):
    src = tmp_path / "export.txt"
    src.write_text(sample_text, encoding="utf-8")

    custom_out = tmp_path / "post_processed.txt"
    llm = FakeLLM()

    out_path_str = clean_transcript_with_llm(input_path=src, output_path=custom_out, llm=llm)
    out_path = Path(out_path_str)

    assert out_path == custom_out
    assert out_path.exists(), "Le fichier custom de sortie doit être créé."
    assert src.exists(), "Le fichier source doit rester intact."


def test_empty_input_creates_empty_like_output(tmp_path: Path):
    src = tmp_path / "empty.txt"
    src.write_text("", encoding="utf-8")

    # Fake LLM renverra aussi vide -> la fonction ne doit pas planter
    llm = FakeLLM(mapping={"": ""})

    out_path_str = clean_transcript_with_llm(input_path=src, output_path=None, llm=llm)
    out_path = Path(out_path_str)

    assert out_path.exists()
    cleaned = out_path.read_text(encoding="utf-8")
    # Dans ce test, on accepte que vide reste vide.
    assert cleaned == "", "Un input vide peut raisonnablement produire un output vide sans erreur."


def test_non_destructive_when_output_exists(tmp_path: Path, sample_text: str):
    """Si un fichier de sortie existe déjà, on vérifie qu'il est bien écrasé
    (comportement courant), OU que la fonction propose une stratégie explicite.
    Ici on teste l'écrasement contrôlé (le plus simple).
    """
    src = tmp_path / "export.txt"
    src.write_text(sample_text, encoding="utf-8")

    out = tmp_path / "test_clean.txt"
    out.write_text("ANCIEN CONTENU", encoding="utf-8")

    llm = FakeLLM()

    out_path_str = clean_transcript_with_llm(input_path=src, output_path=out, llm=llm)
    out_path = Path(out_path_str)

    assert out_path.exists()
    new_content = out_path.read_text(encoding="utf-8")
    assert "ANCIEN CONTENU" not in new_content, "Le fichier de sortie doit être mis à jour."
    assert "d’accord" in new_content
