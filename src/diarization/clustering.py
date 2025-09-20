import logging
import json
import numpy as np
from pathlib import Path

def cluster_speakers(embeddings_file: Path, output_path: Path) -> None:
    """
    Stub for spectral clustering on embeddings.
    Currently just logs and assigns fake speaker IDs.
    """
    logging.info("Clustering speakers from embeddings -> %s", output_path)
    embeddings = np.load(embeddings_file)
    diarization = [{"segment_id": i, "speaker": f"spk{i % 2}"} for i in range(len(embeddings))]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diarization, f)
