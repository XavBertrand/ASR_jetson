from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os
import torch
import gc
from typing import List, Dict, Optional

# === Imports de tes modules existants ===
# Denoise
from asr_jetson.preprocessing.rnnoise import apply_rnnoise as _apply_rnnoise
from asr_jetson.preprocessing.convert_to_wav import convert_to_wav
# VAD
# Diarization
from asr_jetson.diarization.pipeline_diarization import apply_diarization
# ASR
from asr_jetson.asr.whisper_engine import load_faster_whisper
from asr_jetson.asr.transcribe import transcribe_segments, attach_speakers

from asr_jetson.postprocessing.text_export import write_single_block_per_speaker_txt, write_dialogue_txt
from asr_jetson.postprocessing.llm_clean import clean_text_with_llm

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
    denoise: bool = False             # applique RNNoise/afftdn
    device: str = "cpu"              # "cpu" | "cuda"
    n_speakers: int = 2
    clustering_method: str = "hierarchical"      # "spectral" | "kmeans" | "hierarchical" | "ahc_viterbi"
    spectral_assign_labels: str = "kmeans"   # "kmeans" | "cluster_qr"
    vad_min_chunk_s: float = 0.5
    whisper_model: str = "medium"      # tiny/base/small/...
    # Remarque : "int8" = OK CPU seulement ; sur CUDA -> "int8_float16" recommandé
    whisper_compute: str = "int8"      # int8 / int8_float16 / float16 / float32
    language: Optional[str] = None     # None = auto
    out_dir: Path = Path("outputs")    # où écrire JSON/SRT/WAV intermediaire
    diarization_backend: str = "titanet"
    vad_backend: str = "silero"

def _sanitize_whisper_compute(device: str, compute_type: str) -> str:
    """
    Adapte compute_type aux contraintes CTranslate2 :
      - CPU : int8 / float32 ok
      - CUDA : int8 N'EST PAS supporté -> utiliser int8_float16 (ou float16)
    """
    ct = (compute_type or "").lower()
    if device == "cuda":
        if ct in ("int8",):
            return "int8_float16"
        if ct in ("float32", "float16", "int8_float16"):
            return ct
        # fallback raisonnable sur GPU
        return "float16"
    else:
        # CPU : int8/float32 (ou float16 si dispo AVX512fp16, mais inutile ici)
        return ct or "int8"


def run_pipeline(audio_path: str | os.PathLike, cfg: PipelineConfig) -> Dict:
    device = "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"

    # Évite des crashs CTranslate2 : harmonise compute_type selon le device
    compute_type = _sanitize_whisper_compute(device, cfg.whisper_compute)

    # (optionnel) limite threads CT2 pendant les tests CI pour la stabilité
    os.environ.setdefault("CT2_USE_EXPERIMENTAL_PACKED_GEMM", "1")
    os.environ.setdefault("CT2_THREADS", "1")

    src_audio = Path(audio_path)

    # convert to wav
    src_audio = convert_to_wav(src_audio)

    # 0) (optionnel) Denoise -> wav propre
    if cfg.denoise:
        denoised_wav = cfg.out_dir / "intermediate" / (src_audio.stem + "_denoised.wav")
        _ensure_parent(denoised_wav)
        try:
            _apply_rnnoise(src_audio, denoised_wav, model_path=None)
            wav_path = denoised_wav
        except Exception:
            wav_path = src_audio
    else:
        wav_path = src_audio

    # 1) Diarization
    diar_segments = apply_diarization(
        wav_path,
        n_speakers=cfg.n_speakers,
        device=device,
        clustering_method=cfg.clustering_method,
        backend=cfg.diarization_backend,
        vad_backend=cfg.vad_backend,
    )
    if not diar_segments:
        return {"diarization": [], "asr": [], "labeled": []}

    # 2) ASR (faster-whisper)
    model, _meta = load_faster_whisper(
        model_name=cfg.whisper_model,
        device=device,
        compute_type=compute_type,   # <-- utilise la valeur "safe"
    )
    asr_segments = transcribe_segments(
        model, wav_path, diar_segments, language=cfg.language
    )

    # delete whisper model from memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 3) Fusion (ASR + speaker par recouvrement temporel)
    labeled = attach_speakers(diar_segments, asr_segments)
    # labeled: [{start,end,text,speaker}, ...]

    for seg in labeled:
        if "start_s" not in seg:
            seg["start_s"] = float(seg.get("start", 0.0))
        if "end_s" not in seg:
            seg["end_s"] = float(seg.get("end", 0.0))

    # 4) Exports
    _ensure_parent(cfg.out_dir / "json")
    _ensure_parent(cfg.out_dir / "srt")

    root_dir = Path(__file__).resolve().parents[3]
    os.makedirs(os.path.join(root_dir, cfg.out_dir, "json"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, cfg.out_dir, "srt"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, cfg.out_dir, "txt"), exist_ok=True)

    tag = f"_{cfg.vad_backend}_{cfg.diarization_backend}_{cfg.clustering_method}_{cfg.whisper_model}"
    out_json = root_dir / cfg.out_dir / "json" / (Path(audio_path).stem + f"{tag}.json")
    out_srt  = root_dir / cfg.out_dir / "srt" /  (Path(audio_path).stem + f"{tag}.srt")
    out_txt = root_dir / cfg.out_dir / "txt" / (Path(audio_path).stem + f"{tag}.txt")

    # pour le SRT : array avec secondes & string speaker
    srt_payload = [
        {
            "start": seg.get("start_s", seg.get("start", 0.0)),
            "end": seg.get("end_s", seg.get("end", 0.0)),
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

    with open(out_json, "w", encoding="utf-8") as f:
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

    labeled_for_txt = [
        {
            "start": seg.get("start_s", seg.get("start")),
            "end": seg.get("end_s", seg.get("end")),
            "text": seg.get("text", ""),
            "speaker": seg.get("speaker", 0),
        }
        for seg in labeled
    ]

    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # write_single_block_per_speaker_txt(
    #     labeled_for_txt,
    #     out_path=out_txt,
    #     header_style="plain",  # "plain" => SPEAKER_X: ..., "title" => SPEAKER_X (ligne titre)
    # )
    write_dialogue_txt(labeled_for_txt, out_txt)

    # === NOUVEAU : post-correction LLM -> n'écrase pas le .txt original ===

    pp, ff = os.path.split(out_txt)
    ff, ee = os.path.splitext(ff)

    out_txt_clean = root_dir / cfg.out_dir / "txt" / f"{ff}_clean.txt"
    clean_text_with_llm(input_txt=out_txt, output_txt=out_txt_clean)

    return {
        "diarization": diar_segments,
        "asr": asr_segments,
        "labeled": labeled,
        "json": str(out_json),
        "srt": str(out_srt),
        "txt": str(out_txt),
        "txt_llm": str(out_txt_clean),
    }
