# src/asr/whisper_engine.py
from __future__ import annotations
import torch
from typing import Tuple
import os

def load_faster_whisper(model_name: str, device: str, compute_type: str):
    """
    Instancie WhisperModel avec des réglages stables pour les tests/WSL2 :
    - limite le parallélisme (workers/threads=1)
    - conserve le device demandé (GPU si dispo), sinon fallback CPU
    """
    import os
    from faster_whisper import WhisperModel

    os.environ.setdefault("CT2_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=2,   # limite le parallélisme interne
        )
        meta = {"model": model_name, "device": device, "compute_type": compute_type}
        return model, meta
    except Exception as e:
        print(f"[WARN] WhisperModel init failed on device={device} ({e}). Falling back to CPU.")
    except SystemExit:
        print("[WARN] WhisperModel aborted during init. Falling back to CPU.")

    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        num_workers=1,
    )
    meta = {"model": model_name, "device": "cpu", "compute_type": "int8"}
    return model, meta
