import logging
import numpy as np
from pathlib import Path

def extract_speaker_embeddings(audio_path: Path, segments_file: Path, output_path: Path) -> None:
    """
    Stub for TitaNet speaker embedding extraction.
    Currently just logs and writes random embeddings.
    """
    logging.info("Extracting speaker embeddings from %s -> %s", audio_path, output_path)
    fake_embeddings = np.random.rand(5, 256)  # 5 segments, 256-dim embeddings
    np.save(output_path, fake_embeddings)
