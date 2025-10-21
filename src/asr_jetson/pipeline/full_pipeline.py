"""
Full ASR pipeline orchestration, spanning preprocessing, diarization, ASR,
and post-processing exports.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os
import torch
import gc
from typing import Dict, List, Optional, Any

# === Imports from the project modules ===
# Denoising
from asr_jetson.preprocessing.rnnoise import apply_rnnoise as _apply_rnnoise
from asr_jetson.preprocessing.convert_to_wav import convert_to_wav
# Diarization
from asr_jetson.diarization.pipeline_diarization import apply_diarization
# ASR
from asr_jetson.asr.whisper_engine import load_faster_whisper
from asr_jetson.asr.transcribe import transcribe_segments, attach_speakers

from asr_jetson.postprocessing.text_export import write_single_block_per_speaker_txt, write_dialogue_txt
from asr_jetson.postprocessing.llm_clean import clean_text_with_llm
from asr_jetson.postprocessing.anonymizer import Anonymizer


# --- Helpers ---
def _ensure_parent(path: Path) -> None:
    """
    Ensure the parent directory of ``path`` exists.

    :param path: Target path whose parent directory should be created.
    :type path: Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_srt_timestamp(seconds: float) -> str:
    """
    Convert a time value in seconds to an SRT timestamp string.

    :param seconds: Timestamp expressed in seconds.
    :type seconds: float
    :returns: Timestamp formatted as ``HH:MM:SS,mmm``.
    :rtype: str
    """
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(segments: List[Dict[str, Any]], path: Path) -> None:
    """
    Persist segments to an SRT file.

    :param segments: Sequence of segments containing ``start``, ``end``, ``speaker``, and ``text``.
    :type segments: List[Dict[str, Any]]
    :param path: Destination file for the serialized SRT content.
    :type path: Path
    """
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
    """Configuration container governing an ASR pipeline run."""

    denoise: bool = False
    device: str = "cpu"              # "cpu" | "cuda"
    n_speakers: Optional[int] = None
    pyannote_pipeline: str = "pyannote/speaker-diarization-3.1"
    pyannote_auth_token: Optional[str] = None
    whisper_model: str = "medium"      # tiny/base/small/...
    # Note: "int8" is CPU-only; prefer "int8_float16" when targeting CUDA.
    whisper_compute: str = "int8"      # int8 / int8_float16 / float16 / float32
    language: Optional[str] = None     # None = auto-detect
    out_dir: Path = Path("outputs")    # where to write JSON/SRT/WAV intermediates

    anonymize: bool = True
    anon_model: str = "cmarkea/distilcamembert-base-ner"  # lightweight HF NER model
    anon_device: str = "auto"  # "auto" | "cpu" | "cuda"
    anon_catalog: Optional[Path] = None  # JSON/TXT catalog path
    anon_catalog_label: str = "CAT"  # default label when the catalog is plain text/JSON
    anon_catalog_fuzzy: int = 90  # 0 disables fuzzy matching
    anon_catalog_as_person: bool = True  # treat catalog entries as potential PERSON entities


def _sanitize_whisper_compute(device: str, compute_type: str) -> str:
    """
    Normalize the requested Whisper compute type for CTranslate2 compatibility.

    :param device: Execution device string (``"cpu"`` or ``"cuda"``).
    :type device: str
    :param compute_type: Desired CTranslate2 compute type.
    :type compute_type: str
    :returns: Compute type adjusted for the selected device.
    :rtype: str
    """
    ct = (compute_type or "").lower()
    if device == "cuda":
        if ct in ("int8",):
            return "int8_float16"
        if ct in ("float32", "float16", "int8_float16"):
            return ct
        # Reasonable fallback on GPU.
        return "float16"
    else:
        # CPU: int8/float32 (float16 when available, though rarely necessary).
        return ct or "int8"


def run_pipeline(audio_path: str | os.PathLike[str], cfg: PipelineConfig) -> Dict[str, Any]:
    """
    Execute the full ASR pipeline: optional denoising, diarization, ASR decoding,
    speaker assignment, anonymisation, LLM cleanup, and artifact exports.

    :param audio_path: Input audio file path accepted by the pipeline.
    :type audio_path: str | os.PathLike
    :param cfg: Pipeline configuration describing the desired behaviour.
    :type cfg: PipelineConfig
    :returns: Dictionary containing intermediate segments and output artifact paths.
    :rtype: Dict[str, Any]
    """
    device = "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"

    # Avoid CTranslate2 crashes by harmonising compute_type with the device.
    compute_type = _sanitize_whisper_compute(device, cfg.whisper_compute)

    # Optionally limit CT2 threads during CI for stability.
    os.environ.setdefault("CT2_USE_EXPERIMENTAL_PACKED_GEMM", "1")
    os.environ.setdefault("CT2_THREADS", "1")

    src_audio = Path(audio_path)

    # convert to wav
    src_audio = convert_to_wav(src_audio)

    # 0) Optional denoising stage to produce a clean WAV.
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

    # 1) Diarization stage.
    print("=" * 40 + "\n" + "   DIARIZATION\n" + "=" * 40)
    diar_segments = apply_diarization(
        wav_path,
        n_speakers=cfg.n_speakers,
        device=device,
        pyannote_pipeline=cfg.pyannote_pipeline,
        auth_token=cfg.pyannote_auth_token,
    )
    if not diar_segments:
        return {"diarization": [], "asr": [], "labeled": []}

    # 2) ASR with Faster-Whisper.
    print("=" * 40 + "\n" + "   ASR\n" + "=" * 40)
    model, _meta = load_faster_whisper(
        model_name=cfg.whisper_model,
        device=device,
        compute_type=compute_type,   # trusted compute type
    )
    asr_segments = transcribe_segments(
        model, wav_path, diar_segments, language=cfg.language
    )

    # Delete the Whisper model from memory.
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 3) Merge ASR segments and speaker assignments.
    labeled = attach_speakers(diar_segments, asr_segments)
    # labeled: [{start,end,text,speaker}, ...]

    for seg in labeled:
        if "start_s" not in seg:
            seg["start_s"] = float(seg.get("start", 0.0))
        if "end_s" not in seg:
            seg["end_s"] = float(seg.get("end", 0.0))

    # 4) Export artifacts.
    _ensure_parent(cfg.out_dir / "json")
    _ensure_parent(cfg.out_dir / "srt")

    root_dir = Path(__file__).resolve().parents[3]
    os.makedirs(os.path.join(root_dir, cfg.out_dir, "json"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, cfg.out_dir, "srt"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, cfg.out_dir, "txt"), exist_ok=True)

    diar_tag = cfg.pyannote_pipeline.split("/")[-1] if "/" in cfg.pyannote_pipeline else cfg.pyannote_pipeline
    tag = f"_pyannote_{diar_tag}_{cfg.whisper_model}".replace("/", "_")
    out_json = root_dir / cfg.out_dir / "json" / (Path(audio_path).stem + f"{tag}.json")
    out_srt  = root_dir / cfg.out_dir / "srt" /  (Path(audio_path).stem + f"{tag}.srt")
    out_txt = root_dir / cfg.out_dir / "txt" / (Path(audio_path).stem + f"{tag}.txt")

    # Build the SRT payload with second-based timestamps and speaker strings.
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
    #     header_style="plain",  # "plain" => SPEAKER_X: ..., "title" => SPEAKER_X (title line)
    # )
    write_dialogue_txt(labeled_for_txt, out_txt)

    # === POST: Anonymisation -> LLM -> Re-identification ===
    pp, ff = os.path.split(out_txt)
    ff, ee = os.path.splitext(ff)

    out_txt_anon = root_dir / cfg.out_dir / "txt" / f"{ff}_anon.txt"
    out_txt_anon_clean = root_dir / cfg.out_dir / "txt" / f"{ff}_anon_clean.txt"
    out_txt_clean = root_dir / cfg.out_dir / "txt" / f"{ff}_clean.txt"
    out_mapping_json = root_dir / cfg.out_dir / "json" / f"{ff}_anon_mapping.json"

    if cfg.anonymize:
        base_text = out_txt.read_text(encoding="utf-8")

        anonymizer = Anonymizer(model_name=cfg.anon_model, device=cfg.anon_device)
        anon_text, mapping = anonymizer.anonymize(
            base_text,
            catalog_path=str(cfg.anon_catalog) if cfg.anon_catalog else None,
            catalog_label_default=cfg.anon_catalog_label,
            catalog_fuzzy_threshold=int(cfg.anon_catalog_fuzzy),
            catalog_as_person=bool(cfg.anon_catalog_as_person),
        )

        out_txt_anon.write_text(anon_text, encoding="utf-8")
        out_mapping_json.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        # Send the anonymised version to the LLM for clean-up.
        clean_text_with_llm(input_txt=out_txt_anon, output_txt=out_txt_anon_clean)

        # Re-identify (de-anonymise) the LLM-cleaned text.
        anon_clean_text = out_txt_anon_clean.read_text(encoding="utf-8")
        deanonymized = Anonymizer.deanonymize(anon_clean_text, mapping, restore="canonical")
        out_txt_clean.write_text(deanonymized, encoding="utf-8")
    else:
        # Original path without anonymisation.
        clean_text_with_llm(input_txt=out_txt, output_txt=out_txt_clean)

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "diarization": diar_segments,
        "asr": asr_segments,
        "labeled": labeled,
        "json": str(out_json),
        "srt": str(out_srt),
        "txt": str(out_txt),
        "txt_anon": str(out_txt_anon) if cfg.anonymize else None,
        "txt_anon_llm": str(out_txt_anon_clean) if cfg.anonymize else None,
        "txt_llm": str(out_txt_clean),
        "anon_mapping": str(out_mapping_json) if cfg.anonymize else None,
    }
