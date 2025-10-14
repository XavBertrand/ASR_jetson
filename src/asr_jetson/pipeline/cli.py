import argparse
from pathlib import Path
from asr_jetson.pipeline.full_pipeline import PipelineConfig, run_pipeline

def main():
    p = argparse.ArgumentParser(description="Run ASR Jetson pipeline")
    p.add_argument("--audio", required=True, help="Path to input audio (wav/mp3/flac)")
    p.add_argument("--device", default="cuda", help='cpu | cuda (defaults to "cuda")')
    p.add_argument("--speakers", type=int, default=2, help="Expected number of speakers")
    p.add_argument("--whisper-model", default="medium", help="Whisper size (small, large-v3 or h2oai/faster-whisper-large-v3-turbo)")
    p.add_argument("--whisper-compute", default="float16", help="CTranslate2 compute_type")
    p.add_argument("--lang", default="fr", help="Force language code (e.g. fr, en)")
    p.add_argument("--denoise", action="store_true", help="Apply RNNoise/denoise stage")
    p.add_argument("--out-dir", default="outputs", help="Output directory (json/srt/txt)")
    args = p.parse_args()

    cfg = PipelineConfig(
        denoise=args.denoise,
        device=args.device,
        n_speakers=args.speakers,
        whisper_model=args.whisper_model,
        whisper_compute=args.whisper_compute,
        language=args.lang,
        out_dir=Path(args.out_dir),
    )
    result = run_pipeline(args.audio, cfg)
    print("✓ pipeline done\nJSON:", result.get("json"), "\nSRT:", result.get("srt"), "\nTXT:", result.get("txt"), "\nTXT CLEANED:", result.get("txt_llm"))

if __name__ == "__main__":
    main()
