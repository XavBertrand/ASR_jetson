from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Any, List
from mistralai import Mistral

@dataclass
class MistralPrompt:
    model: str
    system: str
    user_prefix: str

def load_prompts(path: str, key: str = "meeting_analysis") -> MistralPrompt:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)[key]
    return MistralPrompt(model=cfg["model"], system=cfg["system"], user_prefix=cfg["user_prefix"])

def chat_complete(model: str, system: str, user_text: str) -> str:
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant dans l'environnement")
    with Mistral(api_key=api_key) as client:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        res = client.chat.complete(model=model, messages=messages, stream=False)
        # Normalisation sortie (SDK varie légèrement selon versions)
        try:
            return res.output[0].content
        except Exception:
            try:
                return res.choices[0].message.content
            except Exception:
                return str(res)
