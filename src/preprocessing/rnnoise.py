import logging
from pathlib import Path

def apply_rnnoise(input_path: Path, output_path: Path) -> None:
    """
    Stub for RNNoise denoising.
    Currently just logs and copies path info.
    """
    logging.info("Running RNNoise denoising on %s -> %s", input_path, output_path)
    # For now: create an empty file as placeholder
    output_path.write_bytes(b"FAKE_WAV_DATA")
