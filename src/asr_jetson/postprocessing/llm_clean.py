# src/postprocessing/llm_clean.py
from __future__ import annotations
import os
from pathlib import Path
import json
import re
import time
import requests
from typing import Optional

os.environ["LLM_ENDPOINT"] = "http://tensorrt-llm:8000"
os.environ["LLM_MODEL"] = "qwen2.5-1.5b-instruct"

SYS_PROMPT = (
    "Tu es un correcteur automatique de transcription française. "
    "Règles: 1) Corrige orthographe, grammaire, accords, ponctuation française (espaces fines avant ; : ! ? facultatives, mais espace normal OK), "
    "2) Garde le sens AU PLUS PROCHE possible, 3) Ne SUPPRIME pas d'information sémantique, "
    "4) Ne réinvente rien, 5) Conserve la structure et les tags de locuteur 'SPEAKER_X:' "
    "6) Fusionne les micro-bégaiements évidents, 7) Ne change pas les numéros de SPEAKER."
    "8) Tu n’écris JA%AIS de commentaires ni d’analyses. "
    "9) Tu produis UNIQUEMENT le texte corrigé, sans aucun ajout ni retrait. "
    "10) Ton rôle est mécanique, comme un correcteur orthographique humain discipliné."
)

USER_INSTR = (
    "IMPORTANT : Ce qui suit N'EST PAS une analyse, pas un résumé, pas une reformulation.\n"
    "Tu n'es PAS critique littéraire, tu es un correcteur automatique.\n"
    "TA SEULE MISSION : corriger les fautes d’orthographe, de grammaire et de ponctuation du texte donné, "
    "sans rien changer d’autre.\n"
    "Règles obligatoires :\n"
    "1️) Ne supprime RIEN.\n"
    "2️) Ne résume RIEN.\n"
    "3️) Ne commente RIEN.\n"
    "4️) Ne réécris PAS le style.\n"
    "5️) Ne traduis PAS.\n"
    "6️) Tu dois garder le même format exact que le texte d’entrée : mêmes lignes, mêmes préfixes 'SPEAKER_X:'.\n"
    "7️) Si tu sors autre chose que le texte corrigé, la sortie sera REJETÉE.\n"
    "Format attendu (exemple) :\n"
    "Entrée : SPEAKER_1: bonjour je suis alle a la plage\n"
    "Sortie : SPEAKER_1: Bonjour, je suis allé à la plage.\n"
    "\n"
    "Voici le texte à corriger (ne réponds QUE par la version corrigée, sans introduction, sans explication) :\n"
)

def _basic_local_cleanup(text: str) -> str:
    """Fallback minimal si aucun LLM dispo (sécurise la pipeline)."""
    import re

    # 1. Réduction basique des espaces
    s = re.sub(r"\s+", " ", text).strip()

    # 2. Ponctuation : espace avant → retirer, espace après → forcer
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:!?])(?=\S)", r"\1 ", s)

    # 3. Apostrophes typographiques
    s = s.replace(" '", " ’").replace("' ", "’ ").replace("'", "’")

    # 4. Majuscule en début de phrase
    s = re.sub(r"(^|[.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), s)

    # 5. SPEAKER tags sur nouvelle ligne + **un seul espace après**
    s = re.sub(r"\s*(SPEAKER_\d+:)\s*", r"\n\1 ", s).strip()

    # 6. Nettoyage des doublons de retours à la ligne
    s = re.sub(r"\n+", "\n", s)

    # 7. Fin de ligne finale
    if not s.endswith("\n"):
        s += "\n"

    return s


def _call_openai_compatible(endpoint: str, api_key: Optional[str], model: str, text: str, timeout_s: int = 120) -> Optional[str]:
    """
    Appelle une API 'OpenAI-compatible' (TensorRT-LLM engine server, vLLM, LM Studio, OpenRouter local, etc.)
    """
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_INSTR + "\n\n" + text},
        ],
        # "temperature": 0.1,
        # "max_tokens": max(512, min(4096, len(text.split()) * 2)),
        "temperature": 0.0,
        "max_tokens": len(text.split()) * 3,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if r.status_code == 200:
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    return None

def _call_ollama(model: str, text: str, timeout_s: int = 120) -> Optional[str]:
    """
    Appelle un serveur Ollama local (http://localhost:11434) si dispo.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_INSTR + "\n\n" + text},
        ],
        "options": {"temperature": 0.1},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    if r.status_code == 200:
        return r.json().get("message", {}).get("content", "").strip()
    return None

def clean_text_with_llm(
    input_txt: Path,
    output_txt: Path,
    default_model: str = "Qwen2.5-1.5B-Instruct",
) -> Path:
    """
    Lit 'input_txt', corrige le texte avec un LLM léger (Qwen 2.5 1.5B instruct), écrit 'output_txt'.
    Backends supportés (dans cet ordre):
      1) API OpenAI-compatible via LLM_ENDPOINT (+ LLM_API_KEY optionnel) et LLM_MODEL,
      2) Ollama local (http://localhost:11434) via OLLAMA_MODEL,
      3) Fallback nettoyage regex local si aucun backend n'est dispo/répondant.

    Variables d'env (optionnelles):
      - LLM_ENDPOINT (ex: http://qwen-api:8000  ou http://localhost:8000)
      - LLM_API_KEY  (si nécessaire, sinon vide accepté)
      - LLM_MODEL    (par défaut: Qwen2.5-1.5B-Instruct)
      - OLLAMA_MODEL (ex: qwen2.5:1.5b-instruct)

    Retourne le path de output_txt.
    """
    input_txt = Path(input_txt)
    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    raw = input_txt.read_text(encoding="utf-8")

    # 1) OpenAI-compatible
    endpoint = os.getenv("LLM_ENDPOINT", "").strip()
    if endpoint:
        model = os.getenv("LLM_MODEL", default_model).strip() or default_model
        api_key = os.getenv("LLM_API_KEY", "").strip() or None
        try:
            out = _call_openai_compatible(endpoint, api_key, model, raw)
            if out:
                output_txt.write_text(out + ("\n" if not out.endswith("\n") else ""), encoding="utf-8")
                return output_txt
        except Exception:
            pass

    # 2) Ollama local
    ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
    if ollama_model:
        try:
            out = _call_ollama(ollama_model, raw)
            if out:
                output_txt.write_text(out + ("\n" if not out.endswith("\n") else ""), encoding="utf-8")
                return output_txt
        except Exception:
            pass

    # 3) Fallback regex minimal (toujours fonctionner)
    cleaned = _basic_local_cleanup(raw)
    output_txt.write_text(cleaned + ("\n" if not cleaned.endswith("\n") else ""), encoding="utf-8")
    return output_txt
