# src/asr_jetson/vad/marblenet.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torchaudio
import numpy as np

from nemo.collections.asr.models import EncDecFrameClassificationModel


def load_marblenet_vad(name: str = "vad_multilingual_frame_marblenet", device: str = "cuda"):
    try:
        model = EncDecFrameClassificationModel.from_pretrained(
            model_name=name, map_location=device
        )
    except FileNotFoundError:
        # Fallback HF v2.0 (souvent nécessaire) ; strict=False pour ignorer loss.weight, etc.
        model = EncDecFrameClassificationModel.from_pretrained(
            model_name="nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0",
            map_location=device,
            strict=False,
        )
    return model.to(device).eval()

@torch.no_grad()
def apply_vad(
    model,
    wav_path: str | Path,
    sample_rate: int = 16000,
    *,
    thr_on: float = 0.55,          # <- hysteresis ON
    thr_off: float = 0.45,         # <- hysteresis OFF
    min_speech_ms: int = 140,
    min_silence_ms: int = 180,
    merge_gap_ms: int = 140,
    pad_ms: int = 100,             # pad_on = pad_off = 100 ms
) -> List[Dict[str, Any]]:
    """
    VAD MarbleNet *frame-level* avec hysteresis + fusion gaps + padding.
    Retourne [{start: sec, end: sec}].
    """
    wav_path = str(wav_path)
    wav, sr = torchaudio.load(wav_path)

    # Mono + resample
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate

    signal = wav.to(model.device, dtype=torch.float32)
    lengths = torch.tensor([signal.shape[1]], device=signal.device, dtype=torch.int64)

    model.eval()
    out = model(input_signal=signal, input_signal_length=lengths)
    logits = out[0] if isinstance(out, (tuple, list)) else out  # (B, T_frames, C=2)
    if logits.dim() == 3:
        probs = torch.softmax(logits, dim=-1)[..., 1]  # (B, T_frames)
    elif logits.dim() == 2:
        probs = torch.sigmoid(logits)
    else:
        raise RuntimeError(f"Dimensions logits inattendues: {tuple(logits.shape)}")

    p = probs[0].detach().float().cpu().numpy()         # (T_frames,)
    T = len(p)
    if T == 0:
        return []

    total_sec = float(signal.shape[1]) / sample_rate
    sec_per_frame = total_sec / max(1, T)

    # --- HYSTERESIS ---
    # Passe en état "speech" si p >= thr_on ; quitte si p < thr_off
    speech_mask = np.zeros(T, dtype=bool)
    in_speech = False
    for i, pi in enumerate(p):
        if in_speech:
            if pi < thr_off:
                in_speech = False
        else:
            if pi >= thr_on:
                in_speech = True
        speech_mask[i] = in_speech

    # --- Regroupement brut avec min_silence ---
    segments = []
    in_seg = False
    start_idx = 0
    last_true = -10**9
    min_silence_frames = int(round((min_silence_ms / 1000.0) / sec_per_frame))
    min_speech_frames  = int(round((min_speech_ms  / 1000.0) / sec_per_frame))

    for i, is_sp in enumerate(speech_mask):
        if is_sp:
            last_true = i
            if not in_seg:
                in_seg = True
                start_idx = i
        else:
            # si on était en parole et que le silence devient assez long, on clôt
            if in_seg and (i - last_true) >= min_silence_frames:
                end_idx = last_true + 1
                if (end_idx - start_idx) >= max(1, min_speech_frames):
                    segments.append({
                        "start": start_idx * sec_per_frame,
                        "end":   end_idx   * sec_per_frame,
                    })
                in_seg = False

    if in_seg:
        end_idx = T
        if (end_idx - start_idx) >= max(1, min_speech_frames):
            segments.append({
                "start": start_idx * sec_per_frame,
                "end":   end_idx   * sec_per_frame,
            })

    # --- Padding ---
    if pad_ms > 0:
        pad_s = pad_ms / 1000.0
        for s in segments:
            s["start"] = max(0.0, s["start"] - pad_s)
            s["end"]   = min(total_sec, s["end"] + pad_s)

    # --- Fusion des gaps courts ---
    merged = []
    if segments:
        segments.sort(key=lambda d: d["start"])
        merged.append(segments[0])
        max_gap = merge_gap_ms / 1000.0
        for s in segments[1:]:
            if s["start"] - merged[-1]["end"] <= max_gap:
                merged[-1]["end"] = max(merged[-1]["end"], s["end"])
            else:
                merged.append(s)
        # filtre durée mini après fusion
        merged = [s for s in merged if (s["end"] - s["start"]) >= (min_speech_ms / 1000.0)]

    return merged
