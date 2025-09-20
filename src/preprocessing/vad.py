import logging
import json
from pathlib import Path

def apply_vad(input_path: Path, output_path: Path) -> None:
    """
    Stub for Silero VAD.
    Currently just logs and writes dummy segments.
    """
    logging.info("Running VAD on %s -> %s", input_path, output_path)
    segments = [{"start": 0.0, "end": 5.0}, {"start": 6.0, "end": 10.0}]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f)
