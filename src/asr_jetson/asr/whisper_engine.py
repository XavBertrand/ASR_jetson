"""Whisper engine utilities for loading Faster-Whisper models."""
from __future__ import annotations
import os
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


def load_faster_whisper(model_name: str, device: str, compute_type: str) -> Tuple["WhisperModel", Dict[str, str]]:
    """
    Instantiate a :class:`faster_whisper.WhisperModel` with conservative defaults.

    The loader restricts the degree of parallelism to keep resource usage stable
    on constrained devices and gracefully falls back to CPU if GPU initialisation
    fails.

    :param model_name: Identifier of the Faster-Whisper model to load.
    :type model_name: str
    :param device: Preferred device string (``"cpu"`` or ``"cuda"``).
    :type device: str
    :param compute_type: CTranslate2 compute type to request.
    :type compute_type: str
    :returns: Tuple containing the model instance and metadata about the load.
    :rtype: Tuple[WhisperModel, Dict[str, str]]
    """
    from faster_whisper import WhisperModel

    os.environ.setdefault("CT2_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    os.environ.setdefault("CT2_USE_MMAP", "1")
    os.environ.setdefault("CT2_BEAM_SIZE", "5")

    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=2,   # keep internal parallelism low
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
