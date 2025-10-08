import argparse
from .core import run_pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="input wav/mp3/flac")
    p.add_argument("--out", default="out/transcript.json")
    p.add_argument("--device", default="cuda")  # "cpu" sur PC si besoin
    p.add_argument("--config", default="configs/dev.yaml")
    args = p.parse_args()
    run_pipeline(audio_path=args.audio, out_path=args.out, device=args.device)
