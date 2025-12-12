"""
Full ASR pipeline orchestration, spanning preprocessing, diarization, ASR,
and post-processing exports.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
import re
import json
import os
import torch
import gc
from typing import Dict, List, Optional, Any, Set

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
from asr_jetson.postprocessing.anonymizer import load_catalog
from asr_jetson.postprocessing.meeting_report import generate_pdf_report
from asr_jetson.postprocessing.transformer_anonymizer import TransformerAnonymizer
from asr_jetson.postprocessing import mistral_client


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


class _GpuMemoryMonitor:
    """Utility that prints CUDA memory usage at key checkpoints."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled) and torch.cuda.is_available()
        if self.enabled:
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                self.enabled = False

    @staticmethod
    def _fmt(num_bytes: int) -> str:
        mib = num_bytes / (1024 ** 2)
        return f"{mib:.1f} MiB"

    def log(self, label: str) -> None:
        if not self.enabled:
            return
        try:
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
        except RuntimeError:
            self.enabled = False
            return
        used = total - free
        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        print(
            f"[GPU MEM] {label}: used {self._fmt(used)} / {self._fmt(total)} "
            f"(max_allocated {self._fmt(max_allocated)}, max_reserved {self._fmt(max_reserved)})"
        )

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
    monitor_gpu_memory: bool = False

    anonymize: bool = True
    anon_model: str = "urchade/gliner_multi_pii-v1"
    anon_device: str = "auto"  # "auto" | "cpu" | "cuda"
    anon_catalog: Optional[Path] = None  # JSON/TXT catalog path
    anon_catalog_label: str = "CAT"  # default label when the catalog is plain text/JSON
    anon_catalog_fuzzy: int = 90  # legacy knob
    anon_catalog_as_person: bool = True  # treat catalog entries as potential PERSON entities
    # LLM clean-up uses environment variables (LLM_ENDPOINT/LLM_MODEL/LLM_API_KEY/USE_OLLAMA).
    anon_enable_llm_qc: bool = True
    anon_max_block_chars: int = 1200
    anon_max_block_sents: int = 5
    generate_meeting_report: bool = True
    meeting_report_prompts: Path = Path("src/asr_jetson/config/mistral_prompts.json")
    meeting_report_prompt_key: str = "entretien_collaborateur"
    presidio_python: Path = Path(".venv-presidio/bin/python")
    speaker_context: Optional[str] = None
    asr_prompt: Optional[str] = None


def _merge_anonymization_mappings(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a secondary anonymisation mapping (e.g. speaker context) into the base mapping
    while avoiding tag collisions.
    """
    if not extra:
        return base

    merged = copy.deepcopy(base) if base else {}
    merged_entities = merged.get("entities") or {}
    if isinstance(merged_entities, list):
        merged_entities = {
            ent.get("tag", f"<ENT_{idx+1}>"): ent for idx, ent in enumerate(merged_entities)
        }
    merged["entities"] = merged_entities

    merged.setdefault("reverse_map", {})
    merged.setdefault("pseudonym_map", {})
    merged.setdefault("pseudonym_reverse_map", {})
    stats = merged.setdefault("stats", {"total": 0, "by_type": {}})
    stats["total"] = stats.get("total", 0)
    stats.setdefault("by_type", {})

    label_counters: Dict[str, int] = {}
    tag_re = re.compile(r"<\s*([A-Za-z]+)\s*_(\d+)\s*>")
    for tag, info in merged_entities.items():
        label = (info.get("label") or info.get("type") or "ENT").upper()
        match = tag_re.match(str(tag))
        idx = int(match.group(2)) if match else 0
        label_counters[label] = max(label_counters.get(label, 0), idx)

    extra_entities = extra.get("entities") or {}
    if isinstance(extra_entities, list):
        extra_items = []
        for ent in extra_entities:
            tag = ent.get("tag")
            if not tag:
                lbl = (ent.get("label") or ent.get("type") or "ENT").upper()
                label_counters[lbl] = label_counters.get(lbl, 0) + 1
                tag = f"<{lbl}_{label_counters[lbl]}>"
            extra_items.append((tag, ent))
    else:
        extra_items = list(extra_entities.items())

    for raw_tag, info in extra_items:
        label = (info.get("label") or info.get("type") or "ENT").upper()
        tag = str(raw_tag)
        if tag in merged_entities:
            label_counters[label] = label_counters.get(label, 0) + 1
            tag = f"<{label}_{label_counters[label]}>"

        info_copy = copy.deepcopy(info)
        info_copy.setdefault("label", label)
        merged_entities[tag] = info_copy

        canonical = info_copy.get("canonical") or (info_copy.get("values") or [None])[0]
        pseudonym = info_copy.get("pseudonym")
        if canonical:
            merged["reverse_map"].setdefault(tag, canonical)
        if pseudonym:
            merged["pseudonym_map"].setdefault(tag, pseudonym)
            merged["pseudonym_reverse_map"].setdefault(pseudonym, canonical or pseudonym)

        stats["total"] += 1
        stats["by_type"][label] = stats["by_type"].get(label, 0) + 1

    merged["entities"] = merged_entities
    return merged


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


def _resolve_transformers_device(device_pref: str) -> int:
    """
    Convert a user-friendly device string into a transformers-compatible device index.
    """
    pref = (device_pref or "auto").strip().lower()
    if pref in {"auto", "cuda", "gpu"}:
        return 0 if torch.cuda.is_available() else -1
    if pref.startswith("cuda:"):
        try:
            idx = int(pref.split(":", 1)[1])
        except ValueError:
            idx = 0
        return idx if torch.cuda.is_available() else -1
    if pref in {"cpu", "mps"}:
        return -1
    try:
        return int(pref)
    except ValueError:
        return 0 if torch.cuda.is_available() else -1


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
    monitor = _GpuMemoryMonitor(cfg.monitor_gpu_memory and device == "cuda")
    monitor.log("pipeline-start")

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
    monitor.log("after-diarization")
    if not diar_segments:
        return {"diarization": [], "asr": [], "labeled": []}

    # 2) ASR with Faster-Whisper.
    print("=" * 40 + "\n" + "   ASR\n" + "=" * 40)
    model, _meta = load_faster_whisper(
        model_name=cfg.whisper_model,
        device=device,
        compute_type=compute_type,   # trusted compute type
    )
    monitor.log("after-whisper-load")
    asr_segments = transcribe_segments(
        model,
        wav_path,
        diar_segments,
        language=cfg.language,
        initial_prompt=cfg.asr_prompt,
    )
    monitor.log("after-transcription")

    # Delete the Whisper model from memory.
    del model
    torch.cuda.empty_cache()
    gc.collect()
    monitor.log("after-whisper-release")

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

    # === POST: Anonymisation / Clean / Report ===
    _, txt_name = os.path.split(out_txt)
    txt_stem, _ = os.path.splitext(txt_name)

    out_txt_anon = root_dir / cfg.out_dir / "txt" / f"{txt_stem}_anon.txt"
    out_txt_anon_clean = root_dir / cfg.out_dir / "txt" / f"{txt_stem}_anon_clean.txt"
    out_txt_clean = root_dir / cfg.out_dir / "txt" / f"{txt_stem}_clean.txt"
    out_mapping_json = root_dir / cfg.out_dir / "json" / f"{txt_stem}_anon_mapping.json"
    speaker_context_hint = (cfg.speaker_context or "").strip() or None
    speaker_context_anon: Optional[str] = None

    report_outputs: Dict[str, Optional[str]] = {
        "report_anonymized_txt": None,
        "report_txt": None,
        "report_docx": None,
        "report_markdown": None,
        "report_pdf": None,
        "report_status": "disabled",
        "report_reason": "Meeting report generation disabled in configuration.",
    }

    base_text = out_txt.read_text(encoding="utf-8")

    if cfg.anonymize:
        domain_entities: Dict[str, List[str]] = {}
        if cfg.anon_catalog:
            catalog_path = Path(cfg.anon_catalog)
            if not catalog_path.is_absolute():
                catalog_path = (root_dir / catalog_path).resolve()
            entries = load_catalog(catalog_path, default_label=cfg.anon_catalog_label)
            tmp: Dict[str, Set[str]] = {}
            for entry in entries:
                pattern = entry.get("pattern") or ""
                if not pattern.strip():
                    continue
                label = entry.get("label") or cfg.anon_catalog_label
                if cfg.anon_catalog_as_person:
                    label = "PERSON"
                normalized = TransformerAnonymizer._normalize_type(label)
                tmp.setdefault(normalized, set()).add(pattern.strip())
            domain_entities = {label: sorted(values) for label, values in tmp.items() if values}

        anonymizer = TransformerAnonymizer(
            model_name=cfg.anon_model,
            domain_entities=domain_entities or None,
            device=_resolve_transformers_device(cfg.anon_device),
        )
        anonymized_text, mapping = anonymizer.anonymize_with_tags(base_text)

        if speaker_context_hint:
            context_anonymized, context_mapping = anonymizer.anonymize_with_tags(speaker_context_hint)
            mapping = _merge_anonymization_mappings(mapping, context_mapping)
            speaker_context_anon = context_anonymized

        corrected_text = mapping.get("corrected_text")
        if isinstance(corrected_text, str) and corrected_text and corrected_text != base_text:
            base_text = corrected_text
            out_txt.write_text(base_text, encoding="utf-8")
        out_txt_clean.write_text(base_text, encoding="utf-8")

        out_txt_anon.write_text(anonymized_text, encoding="utf-8")

        if cfg.anon_enable_llm_qc:
            try:
                clean_text_with_llm(out_txt_anon, out_txt_anon_clean)
            except Exception as exc:  # pragma: no cover - depends on external services
                print(f"⚠️ LLM clean-up failed: {exc}")
                out_txt_anon_clean.write_text(anonymized_text, encoding="utf-8")
        else:
            out_txt_anon_clean.write_text(anonymized_text, encoding="utf-8")

        out_mapping_json.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    else:
        # anonymize=False -> on copie juste le texte brut dans les sorties "clean"
        out_txt_anon.write_text(base_text, encoding="utf-8")
        out_txt_anon_clean.write_text(base_text, encoding="utf-8")
        out_mapping_json.write_text("{}", encoding="utf-8")
        out_txt_clean.write_text(base_text, encoding="utf-8")
        if speaker_context_hint:
            speaker_context_anon = speaker_context_hint

    if cfg.generate_meeting_report:
        prompts_path = cfg.meeting_report_prompts
        if not prompts_path.is_absolute():
            prompts_path = (root_dir / prompts_path).resolve()
        if not prompts_path.exists():
            raise FileNotFoundError(f"Mistral prompts file not found: {prompts_path}")

        prompt = mistral_client.load_prompts(str(prompts_path), key=cfg.meeting_report_prompt_key)
        anonymized_payload = out_txt_anon_clean.read_text(encoding="utf-8")
        if speaker_context_anon:
            anonymized_payload = (
                f"Contexte sur les interlocuteurs (anonymisé) :\n{speaker_context_anon}\n\n"
                + anonymized_payload
            )
        analysis_anonymized = mistral_client.chat_complete(
            model=prompt.model,
            system=prompt.system,
            user_text=prompt.user_prefix + anonymized_payload,
            temperature=0.1,
        )

        reports_dir = root_dir / cfg.out_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_anon_path = reports_dir / f"{txt_stem}_meeting_report_anonymized.md"
        report_anon_path.write_text(analysis_anonymized, encoding="utf-8")

        report_outputs = generate_pdf_report(
            anonymized_markdown_path=report_anon_path,
            mapping_json_path=out_mapping_json,
            output_dir=root_dir / cfg.out_dir,
            run_id=txt_stem,
            prompt_key=cfg.meeting_report_prompt_key,
        )
        # Preserve legacy keys expected by callers.
        report_outputs.setdefault("report_anonymized_txt", str(report_anon_path))
        report_outputs.setdefault("report_txt", None)
        report_outputs.setdefault("report_docx", None)

    torch.cuda.empty_cache()
    gc.collect()
    monitor.log("pipeline-end")

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
        **report_outputs,
    }
