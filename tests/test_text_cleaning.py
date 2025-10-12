# tests/test_llm_clean.py
import os
from pathlib import Path
import json
import types
import pytest

# Import de la fonction à tester
from asr_jetson.postprocessing.llm_clean import clean_text_with_llm

# ---------------------------------------------------------------------
# Helpers de mocks pour requests.post
# ---------------------------------------------------------------------

class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _mock_post_openai_ok(url, headers=None, json=None, timeout=None):
    # Simule /v1/chat/completions
    assert url.endswith("/v1/chat/completions")
    # On peut vérifier que le prompt est bien construit si besoin
    assert isinstance(json, dict) and "messages" in json
    payload = {
        "choices": [
            {"message": {"content": "SPEAKER_0: Bonjour, je m’appelle Xavier.\nSPEAKER_1: Enchanté !"}}
        ]
    }
    return _Resp(200, payload)


def _mock_post_ollama_ok(url, json=None, timeout=None):
    # Simule http://localhost:11434/api/chat
    assert "http://localhost:11434/api/chat" in url
    payload = {
        "message": {"content": "SPEAKER_0: Bonjour, je m’appelle Xavier.\nSPEAKER_1: Enchanté !"}
    }
    return _Resp(200, payload)


def _mock_post_fail(url, *args, **kwargs):
    # Renvoie un statut non-200 pour forcer le fallback
    return _Resp(500, {})


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_clean_text_with_llm_openai_compatible(tmp_path, monkeypatch):
    """
    Chemin 1 : API OpenAI-compatible via LLM_ENDPOINT.
    """
    # Arrange (input)
    raw = "SPEAKER_0: bonjour ,je  m'appelle  'xavier'! SPEAKER_1: enchante !"
    in_file = tmp_path / "in.txt"
    out_file = tmp_path / "out.txt"
    in_file.write_text(raw, encoding="utf-8")

    # Env pour déclencher le chemin OpenAI-compatible
    monkeypatch.setenv("LLM_ENDPOINT", "http://localhost:8000")
    monkeypatch.setenv("LLM_MODEL", "Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("LLM_API_KEY", "dummy")

    # Mock de requests.post
    import requests
    monkeypatch.setattr(requests, "post", _mock_post_openai_ok)

    # Act
    result_path = clean_text_with_llm(in_file, out_file)

    # Assert
    out = result_path.read_text(encoding="utf-8")
    assert result_path == out_file
    assert out.endswith("\n")
    # Le contenu doit être exactement la réponse mockée
    assert "SPEAKER_0: Bonjour, je m’appelle Xavier." in out
    assert "SPEAKER_1: Enchanté !" in out


# def test_clean_text_with_llm_ollama(tmp_path, monkeypatch):
#     """
#     Chemin 2 : Ollama local si LLM_ENDPOINT n'est pas fourni.
#     """
#     raw = "SPEAKER_0: salut ,je  m'appelle  'xav'!"
#     in_file = tmp_path / "in.txt"
#     out_file = tmp_path / "out.txt"
#     in_file.write_text(raw, encoding="utf-8")
#
#     # Aucune API OpenAI-compatible
#     monkeypatch.delenv("LLM_ENDPOINT", raising=False)
#     monkeypatch.delenv("LLM_API_KEY", raising=False)
#     monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
#
#     # Mock de requests.post (Ollama)
#     import requests
#     monkeypatch.setattr(requests, "post", _mock_post_ollama_ok)
#
#     # Act
#     result_path = clean_text_with_llm(in_file, out_file)
#
#     # Assert
#     out = result_path.read_text(encoding="utf-8")
#     assert "SPEAKER_0: Bonjour, je m’appelle Xavier." in out
#     assert out.endswith("\n")


def test_clean_text_with_llm_fallback_regex(tmp_path, monkeypatch):
    """
    Chemin 3 : Fallback regex lorsque aucun backend ne répond.
    """
    raw = "SPEAKER_0: bonjour  ,je  m'appelle  'xav'!  SPEAKER_1:  ca  va?"
    in_file = tmp_path / "in.txt"
    out_file = tmp_path / "out.txt"
    in_file.write_text(raw, encoding="utf-8")

    # Pas d'endpoint LLM / Ollama
    monkeypatch.delenv("LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    # Mock requests.post qui échoue (au cas où)
    import requests
    monkeypatch.setattr(requests, "post", _mock_post_fail)

    # Act
    result_path = clean_text_with_llm(in_file, out_file)

    # Assert (ne teste pas le texte exact, juste des invariants du cleanup)
    out = result_path.read_text(encoding="utf-8")
    assert result_path == out_file
    assert out.endswith("\n")
    # Le fallback conserve les tags SPEAKER et réduit les espaces superflus
    assert "SPEAKER_0:" in out and "SPEAKER_1:" in out
    # Pas de doubles espaces
    assert "  " not in out
    # Apostrophes typographiques probables
    assert "’" in out or "' " not in out  # assez souple selon l’OS/locale
