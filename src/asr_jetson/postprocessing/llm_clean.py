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

FEW_SHOT_EXAMPLES = """
EXEMPLE 1:
Entrée:
SPEAKER_1: bonjour je mappel jean je suis content detre ici
SPEAKER_2: enchanté moi cest marie

Sortie:
SPEAKER_1: Bonjour, je m'appelle Jean. Je suis content d'être ici.
SPEAKER_2: Enchanté, moi c'est Marie.

EXEMPLE 2:
Entrée:
SPEAKER_1: decleration des droit de lhomme et du citoyen
SPEAKER_2: oui cest un texte fondementale

Sortie:
SPEAKER_1: Déclaration des droits de l'homme et du citoyen.
SPEAKER_2: Oui, c'est un texte fondamental.
"""

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
    "Ne réponds QUE par la version corrigée, sans introduction, sans explication.\n"
    + FEW_SHOT_EXAMPLES +
    "MAINTENANT TON TOUR, voici le texte à corriger :\n"
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

# def _call_ollama(model: str, text: str, timeout_s: int = 120) -> Optional[str]:
#     """
#     Appelle un serveur Ollama local (http://localhost:11434) si dispo.
#     """
#     url = "http://localhost:11434/api/chat"
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": SYS_PROMPT},
#             {"role": "user", "content": USER_INSTR + "\n\n" + text},
#         ],
#         "options": {"temperature": 0.1},
#         "stream": False,
#     }
#     r = requests.post(url, json=payload, timeout=timeout_s)
#     if r.status_code == 200:
#         return r.json().get("message", {}).get("content", "").strip()
#     return None
def _call_ollama(model: str, text: str, timeout_s: int = 120) -> Optional[str]:
    """
    Appelle un serveur Ollama local avec contraintes strictes.
    Utilise un message "assistant" préfixé pour forcer le format.
    """
    url = "http://localhost:11434/api/chat"

    # Extraire le premier tag SPEAKER
    first_speaker_match = re.match(r'(SPEAKER_\d+:)', text)
    first_speaker = first_speaker_match.group(1) if first_speaker_match else "SPEAKER_1:"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_INSTR + "\n\n" + text},
            # Astuce : on "amorce" la réponse du modèle
            {"role": "assistant", "content": first_speaker + " "},
        ],
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": len(text.split()) * 3,
        },
        "stream": False,
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code == 200:
            content = r.json().get("message", {}).get("content", "").strip()

            # Le modèle va continuer après notre préfixe
            # On reconstruit la réponse complète
            full_content = first_speaker + " " + content

            # Validation
            if not _validate_ollama_output(text, full_content):
                return None

            return full_content
        else:
            print(f"⚠️ Ollama erreur HTTP {r.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Erreur Ollama: {e}")
        return None


def _validate_ollama_output(original: str, corrected: str) -> bool:
    """Valide que Ollama a bien corrigé et non analysé."""

    # 1. Vérifier les SPEAKER tags
    orig_speakers = set(re.findall(r'SPEAKER_\d+:', original))
    corr_speakers = set(re.findall(r'SPEAKER_\d+:', corrected))
    if orig_speakers != corr_speakers:
        print(f"⚠️ SPEAKER tags différents: {orig_speakers} vs {corr_speakers}")
        return False

    # 2. Vérifier la longueur (doit rester similaire ±50%)
    orig_words = len(original.split())
    corr_words = len(corrected.split())
    if corr_words < orig_words * 0.5 or corr_words > orig_words * 1.5:
        print(f"⚠️ Longueur suspecte: {orig_words} mots → {corr_words} mots")
        return False

    # 3. Détecter si c'est une analyse (mots interdits au début)
    forbidden_start = ["voici", "cette", "le texte", "analyse", "résumé", "thème"]
    first_line = corrected.split('\n')[0].lower()
    if any(first_line.startswith(word) for word in forbidden_start):
        print(f"⚠️ Détection d'analyse: {first_line[:100]}")
        return False

    # 4. Vérifier qu'il n'y a pas de structure d'analyse (numéros, tirets)
    if re.search(r'^\d+\.', corrected, re.MULTILINE) or corrected.count('-') > orig_words * 0.1:
        print(f"⚠️ Structure d'analyse détectée")
        return False

    return True

def clean_text_with_llm(
    input_txt: Path,
    output_txt: Path,
    default_model: str = "Qwen2.5-1.5B-Instruct",
) -> Path:
    input_txt = Path(input_txt)
    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    raw = input_txt.read_text(encoding="utf-8")

    # 1) OpenAI-compatible (garde ton code existant)
    endpoint = os.getenv("LLM_ENDPOINT", "").strip()
    if endpoint:
        model = os.getenv("LLM_MODEL", default_model).strip() or default_model
        api_key = os.getenv("LLM_API_KEY", "").strip() or None
        try:
            out = _call_openai_compatible(endpoint, api_key, model, raw)
            if out and _validate_ollama_output(raw, out):
                output_txt.write_text(out + "\n", encoding="utf-8")
                print("✅ Correction LLM (OpenAI-compatible) OK")
                return output_txt
            else:
                print("⚠️ Sortie OpenAI-compatible invalide")
        except Exception as e:
            print(f"⚠️ Erreur OpenAI-compatible: {e}")

    # 2) Ollama local (VERSION AMÉLIORÉE)
    ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
    if ollama_model:
        print(f"🔄 Tentative Ollama avec modèle: {ollama_model}")
        try:
            out = _call_ollama(ollama_model, raw)
            if out:
                output_txt.write_text(out + "\n", encoding="utf-8")
                print("✅ Correction Ollama OK")
                return output_txt
            else:
                print("⚠️ Sortie Ollama invalide, fallback regex")
        except Exception as e:
            print(f"⚠️ Erreur Ollama: {e}")

    # 3) Fallback regex minimal (toujours fonctionner)
    print("🔧 Utilisation du fallback regex")
    cleaned = _basic_local_cleanup(raw)
    output_txt.write_text(cleaned + "\n", encoding="utf-8")
    return output_txt
