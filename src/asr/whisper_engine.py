# src/asr/whisper_engine.py
from __future__ import annotations
import torch
from typing import Tuple
import os

def load_faster_whisper(model_name: str = "tiny", device: str = "cpu", compute_type: str = "int8") -> Tuple[object, dict]:
    """
    Charge le modèle faster-whisper.
    - model_name: tiny | base | small | medium | large-v3 (pour CI : tiny/base conseillés)
    - device: "cpu" | "cuda"
    - compute_type: "int8"/"int8_float16"/"float16"/"float32"
    """
    try:
        from faster_whisper import WhisperModel  # pip install faster-whisper
    except Exception as e:
        raise RuntimeError(f"faster-whisper indisponible : {e}")

    device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() != "" else "cpu"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return model, {"device": device, "compute_type": compute_type}
