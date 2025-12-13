from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Any, List

try:
    from mistralai import Mistral  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Mistral = None  # type: ignore

@dataclass
class MistralPrompt:
    model: str
    system: str
    user_prefix: str

def load_prompts(path: str, key: str = "meeting_analysis") -> MistralPrompt:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)[key]
    return MistralPrompt(model=cfg["model"], system=cfg["system"], user_prefix=cfg["user_prefix"])

def chat_complete(
    model: str,
    system: str,
    user_text: str,
    temperature: float | None = None,
) -> str:
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant dans l'environnement")
    if Mistral is None:
        raise RuntimeError("Le package 'mistralai' est requis pour utiliser l'API Mistral.")
    with Mistral(api_key=api_key) as client:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        params: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if temperature is not None:
            params["temperature"] = temperature
        res = client.chat.complete(**params)
        # Normalisation sortie (SDK varie légèrement selon versions)
        try:
            return res.output[0].content
        except Exception:
            try:
                return res.choices[0].message.content
            except Exception:
                return str(res)
