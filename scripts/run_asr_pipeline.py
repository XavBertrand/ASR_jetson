#!/usr/bin/env python
import argparse
from pathlib import Path
from src.pipeline.full_pipeline import PipelineConfig, run_pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=str)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ns", "--n_speakers", type=int, default=2)
    p.add_argument("--whisper", default="tiny")
    p.add_argument("--compute", default="int8")
    p.add_argument("--lang", default=None)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    cfg = PipelineConfig(
        denoise=bool(args.denoise),
        device=args.device,
        n_speakers=args.ns,
        whisper_model=args.whisper,
        whisper_compute=args.compute,
        language=args.lang,
        out_dir=Path(args.out),
    )
    res = run_pipeline(args.audio, cfg)
    print(f"OK — JSON: {res.get('json')}  SRT: {res.get('srt')}")

if __name__ == "__main__":
    main()
