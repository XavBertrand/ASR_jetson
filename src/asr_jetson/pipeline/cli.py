"""
Command-line entry point for orchestrating the ASR Jetson pipeline.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

from asr_jetson.pipeline.full_pipeline import (
    PipelineConfig,
    _build_run_id,
    _resolve_out_root,
    _write_manifest,
    run_pipeline,
)

_DEFAULT_REPORT_TYPE = "entretien_collaborateur"
_DEFAULT_ASR_PROMPT = "Kleos, Pennylane, CJD"


def _resolve_meta_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _meta_dir_is_run_root(meta_dir: Path) -> bool:
    return any((meta_dir / name).exists() for name in ("txt", "json", "reports", "srt"))


def _resolve_run_root(meta: dict[str, Any], meta_dir: Path) -> Path:
    if _meta_dir_is_run_root(meta_dir):
        return meta_dir
    run_root_value = meta.get("run_root")
    if isinstance(run_root_value, str) and run_root_value.strip():
        run_root = _resolve_meta_path(meta_dir, run_root_value)
        if run_root.exists():
            return run_root
    return meta_dir


def _resolve_audio_path(
    meta: dict[str, Any],
    meta_dir: Path,
    run_root: Path,
) -> Optional[Path]:
    for key in ("audio_path", "saved_path", "audio"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            candidate = _resolve_meta_path(meta_dir, value)
            if candidate.exists():
                return candidate
            if not Path(value).is_absolute():
                 return candidate

    for key in ("saved_filename", "original_filename"):
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidates = [
            meta_dir / value,
            run_root / value,
            run_root.parent / value,
            run_root.parent.parent / value,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return None


def main() -> None:
    """
    Parse CLI arguments and execute a full ASR pipeline run.

    :returns: ``None``. The function exits after printing output file paths.
    :rtype: None
    """
    p = argparse.ArgumentParser(description="Run ASR Jetson pipeline")
    p.add_argument("--audio", required=False, help="Path to input audio (wav/mp3/flac)")
    p.add_argument("--device", default="cuda", help='cpu | cuda (defaults to "cuda")')
    p.add_argument("--speakers", type=int, default=None, help="Optional expected number of speakers")
    p.add_argument("--whisper-model", default="large-v3", help="Whisper size (small, large-v3 or h2oai/faster-whisper-large-v3-turbo or openai/whisper-large-v3-turbo)")
    p.add_argument("--whisper-compute", default="int8_float16", help="CTranslate2 compute_type")
    p.add_argument("--lang", default="fr", help="Force language code (e.g. fr, en)")
    p.add_argument("--denoise", action="store_true", help="Apply RNNoise/denoise stage")
    p.add_argument("--out-dir", default="outputs", help="Output directory (json/srt/txt)")
    p.add_argument("--pyannote-pipeline", default="pyannote/speaker-diarization-3.1", help="Pyannote pipeline identifier to use for diarization")
    p.add_argument("--pyannote-token", default=None, help="Hugging Face token for private Pyannote pipelines (optional)")
    p.add_argument("--monitor-gpu-memory", action="store_true",help="Print GPU memory usage at key stages of the pipeline")
    p.add_argument(
        "--asr-prompt",
        type=str,
        default=_DEFAULT_ASR_PROMPT,
        help="Optional initial prompt sent to Faster-Whisper to bias decoding",
    )
    p.add_argument("--speaker-context", type=str, default=None, help="Optional anonymized description of the speakers/roles to help the report (kept local)")
    p.add_argument("--run-id", type=str, default=None, help="Optional run identifier to group outputs")
    p.add_argument("--recordings-root", type=str, default=None, help="Optional recordings root to relativize manifest paths")
    p.add_argument("--report-only", action="store_true", help="Only regenerate the anonymization/report using existing transcripts")
    p.add_argument(
        "--meta-json",
        type=str,
        default=None,
        help="Path to meta.json (overrides meeting date/type/context; relative paths resolved from its folder).",
    )
    p.add_argument(
        "--meeting-date",
        type=str,
        default=None,
        help="Date de l'entretien (YYYY-MM-DD) utilisée pour le prompt et le nom du rapport (défaut : aujourd'hui)",
    )
    p.add_argument("--meeting-report-type",
        type=str,
        default=_DEFAULT_REPORT_TYPE,
        choices=[
            "entretien_collaborateur",
            "entretien_client_particulier_contentieux",
            "entretien_client_professionnel_conseil",
            "entretien_client_professionnel_contentieux",
            "compte_rendu_association",

        ],
        help="Prompt category for the meeting report (matches keys in mistral_prompts.json)",
    )
    args = p.parse_args()

    if args.meta_json:
        meta_path = Path(args.meta_json).expanduser().resolve()
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_dir = meta_path.parent
        run_root_from_meta = _resolve_run_root(meta_data, meta_dir)

        if args.report_only:
            args.out_dir = str(run_root_from_meta)

        meeting_date = meta_data.get("meeting_date")
        if meeting_date and not args.meeting_date:
            args.meeting_date = meeting_date

        report_type = meta_data.get("meeting_report_type")
        if report_type and args.meeting_report_type == _DEFAULT_REPORT_TYPE:
            args.meeting_report_type = report_type

        speaker_context = meta_data.get("speaker_context")
        if speaker_context and not args.speaker_context:
            args.speaker_context = speaker_context

        asr_prompt = meta_data.get("asr_prompt")
        if asr_prompt and args.asr_prompt == _DEFAULT_ASR_PROMPT:
            args.asr_prompt = asr_prompt

        run_id = meta_data.get("run_id")
        if run_id and not args.run_id:
            args.run_id = run_id

        if not args.audio:
            audio_path = _resolve_audio_path(meta_data, meta_dir, run_root_from_meta)
            if audio_path:
                args.audio = str(audio_path)

    if not args.audio:
        raise ValueError("--audio is required unless meta.json provides an audio path.")
    recordings_root = args.recordings_root or os.environ.get("ASR_RECORDINGS_ROOT")

    cfg = PipelineConfig(
        denoise=args.denoise,
        device=args.device,
        n_speakers=args.speakers,
        whisper_model=args.whisper_model,
        whisper_compute=args.whisper_compute,
        language=args.lang,
        out_dir=Path(args.out_dir),
        pyannote_pipeline=args.pyannote_pipeline,
        pyannote_auth_token=args.pyannote_token,
        monitor_gpu_memory=args.monitor_gpu_memory,
        asr_prompt=args.asr_prompt,
        speaker_context=args.speaker_context,
        meeting_report_prompt_key=args.meeting_report_type,
        meeting_date=args.meeting_date,
        run_id=args.run_id,
        recordings_root=Path(recordings_root) if recordings_root else None,
        report_only=args.report_only,
    )
    try:
        result = run_pipeline(args.audio, cfg)
    except Exception as exc:
        run_id = cfg.run_id or _build_run_id(Path(args.audio).stem)
        try:
            _write_manifest(
                _resolve_out_root(cfg),
                run_id=run_id,
                status="failed",
                audio_path=Path(args.audio),
                cfg=cfg,
                report_outputs=None,
                error=str(exc),
            )
        except Exception as manifest_exc:
            print(f"[WARN] Manifest write failed: {manifest_exc}")
        raise
    print(
        "✓ pipeline done\nJSON:",
        result.get("json"),
        "\nSRT:",
        result.get("srt"),
        "\nTXT:",
        result.get("txt"),
        "\nTXT CLEANED:",
        result.get("txt_llm"),
    )

if __name__ == "__main__":
    main()
