from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os
import torch
from typing import List, Dict, Optional

# === Imports de tes modules existants ===
# Denoise
from src.preprocessing.rnnoise import apply_rnnoise as _apply_rnnoise
# VAD
from src.preprocessing.vad import load_silero_vad, apply_vad
# Diarization
from src.diarization.pipeline_diarization import apply_diarization
# ASR
from src.asr.whisper_engine import load_faster_whisper
from src.asr.transcribe import transcribe_segments, attach_speakers

# --- Helpers ---
def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _format_srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _write_srt(segments: List[Dict], path: Path):
    _ensure_parent(path)
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_timestamp(seg["start"])
        end = _format_srt_timestamp(seg["end"])
        spk = seg.get("speaker", "SPEAKER")
        text = seg.get("text", "").strip()
        lines += [
            f"{i}",
            f"{start} --> {end}",
            f"{spk}: {text}" if text else f"{spk}",
            ""
        ]
    path.write_text("\n".join(lines), encoding="utf-8")

@dataclass
class PipelineConfig:
    denoise: bool = True             # applique RNNoise/afftdn
    device: str = "cuda"              # "cpu" | "cuda"
    n_speakers: int = 2
    clustering_method: str = "spectral"      # "spectral" | "kmeans"
    spectral_assign_labels: str = "kmeans"   # "kmeans" | "cluster_qr"
    vad_min_chunk_s: float = 0.5
    whisper_model: str = "tiny"      # tiny/base/small/...
    whisper_compute: str = "int8"    # int8/int8_float16/float16/float32
    language: Optional[str] = None   # None = auto
    out_dir: Path = Path("outputs")  # où écrire JSON/SRT/WAV intermediaire

def run_pipeline(audio_path: str | os.PathLike, cfg: PipelineConfig) -> Dict:
    device = "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    src_audio = Path(audio_path)

    # 0) (optionnel) Denoise -> wav propre
    if cfg.denoise:
        denoised_wav = cfg.out_dir / "intermediate" / (src_audio.stem + "_denoised.wav")
        _ensure_parent(denoised_wav)
        try:
            _apply_rnnoise(src_audio, denoised_wav, model_path=None)  # ton wrapper gère arnndn/afftdn fallback
            wav_path = denoised_wav
        except Exception:
            # en cas d’échec, on retombe sur l’input brut
            wav_path = src_audio
    else:
        wav_path = src_audio

    # 1) Diarization (fait déjà : VAD (Silero) -> TitaNet -> clustering)
    diar_segments = apply_diarization(
        wav_path,
        n_speakers=cfg.n_speakers,
        device=device,
        clustering_method=cfg.clustering_method,
    )  # -> [{start,end,start_s,end_s,speaker}, ...]
    if not diar_segments:
        return {"diarization": [], "asr": [], "labeled": []}

    # 2) ASR (faster-whisper)
    model, _meta = load_faster_whisper(
        model_name=cfg.whisper_model, device=device, compute_type=cfg.whisper_compute
    )
    asr_segments = transcribe_segments(
        model, wav_path, diar_segments, language=cfg.language
    )  # -> [{start,end,text}, ...]

    # 3) Fusion (ASR + speaker par recouvrement temporel)
    labeled = attach_speakers(diar_segments, asr_segments)
    # labeled: [{start,end,text,speaker}, ...]

    # 4) Exports
    _ensure_parent(cfg.out_dir / "json")
    _ensure_parent(cfg.out_dir / "srt")

    out_json = cfg.out_dir / "json" / (Path(audio_path).stem + ".json")
    out_srt  = cfg.out_dir / "srt" /  (Path(audio_path).stem + ".srt")

    # pour le SRT : array avec secondes & string speaker
    srt_payload = [
        {
            "start": seg["start_s"],
            "end": seg["end_s"],
            "speaker": f"SPK{seg.get('speaker', 0)}",
            "text": seg.get("text", ""),
        }
        for seg in labeled
    ]
    _write_srt(
        [
            {
                "start": s["start"],
                "end": s["end"],
                "speaker": s["speaker"],
                "text": s["text"],
            }
            for s in srt_payload
        ],
        out_srt,
    )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "audio": str(Path(audio_path).resolve()),
                "device": device,
                "n_speakers": cfg.n_speakers,
                "segments": labeled,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "diarization": diar_segments,
        "asr": asr_segments,
        "labeled": labeled,
        "json": str(out_json),
        "srt": str(out_srt),
    }
