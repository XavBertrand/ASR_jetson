"""
Command-line entry point for orchestrating the ASR Jetson pipeline.
"""
import argparse
from pathlib import Path

from asr_jetson.pipeline.full_pipeline import PipelineConfig, run_pipeline


def main() -> None:
    """
    Parse CLI arguments and execute a full ASR pipeline run.

    :returns: ``None``. The function exits after printing output file paths.
    :rtype: None
    """
    p = argparse.ArgumentParser(description="Run ASR Jetson pipeline")
    p.add_argument("--audio", required=True, help="Path to input audio (wav/mp3/flac)")
    p.add_argument("--device", default="cuda", help='cpu | cuda (defaults to "cuda")')
    p.add_argument("--speakers", type=int, default=None, help="Optional expected number of speakers")
    p.add_argument("--whisper-model", default="h2oai/faster-whisper-large-v3-turbo", help="Whisper size (small, large-v3 or h2oai/faster-whisper-large-v3-turbo or openai/whisper-large-v3-turbo)")
    p.add_argument("--whisper-compute", default="int8_float16", help="CTranslate2 compute_type")
    p.add_argument("--lang", default="fr", help="Force language code (e.g. fr, en)")
    p.add_argument("--denoise", action="store_true", help="Apply RNNoise/denoise stage")
    p.add_argument("--out-dir", default="outputs", help="Output directory (json/srt/txt)")
    p.add_argument("--pyannote-pipeline", default="pyannote/speaker-diarization-3.1", help="Pyannote pipeline identifier to use for diarization")
    p.add_argument("--pyannote-token", default=None, help="Hugging Face token for private Pyannote pipelines (optional)")
    p.add_argument("--monitor-gpu-memory", action="store_true",help="Print GPU memory usage at key stages of the pipeline")
    p.add_argument("--asr-prompt", type=str, default="Kleos, Pennylane, CJD, Manupro, El Moussaoui", help="Optional initial prompt sent to Faster-Whisper to bias decoding")
    p.add_argument("--speaker-context", type=str, default=None, help="Optional anonymized description of the speakers/roles to help the report (kept local)")
    p.add_argument(
        "--meeting-date",
        type=str,
        default=None,
        help="Date de l'entretien (YYYY-MM-DD) utilisée pour le prompt et le nom du rapport (défaut : aujourd'hui)",
    )
    p.add_argument("--meeting-report-type",
        type=str,
        default="entretien_collaborateur",
        choices=[
            "entretien_collaborateur",
            "entretien_client_particulier_contentieux",
            "entretien_client_professionnel_conseil",
            "entretien_client_professionnel_contentieux",
        ],
        help="Prompt category for the meeting report (matches keys in mistral_prompts.json)",
    )
    args = p.parse_args()

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
    )
    result = run_pipeline(args.audio, cfg)
    print("✓ pipeline done\nJSON:", result.get("json"), "\nSRT:", result.get("srt"), "\nTXT:", result.get("txt"), "\nTXT CLEANED:", result.get("txt_llm"))

if __name__ == "__main__":
    main()
