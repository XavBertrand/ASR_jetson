import subprocess
import shutil
import logging
from pathlib import Path

def apply_rnnoise(input_path: Path, output_path: Path, model_path: Path = None, filter_type: str = "arnndn") -> None:
    """
    Apply noise reduction using ffmpeg (arnndn if available, otherwise afftdn).

    Args:
        input_path (Path): Input audio file
        output_path (Path): Path where the denoised wav will be saved
        model_path (Path, optional): Path to RNNoise .nn model (only for arnndn)
        filter_type (str): "arnndn" (with model) or "afftdn" (fallback)

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if filter_type == "arnndn":
        if model_path and model_path.exists():
            filter_arg = f"arnndn=m='{str(model_path)}':mix=0.9"
        else:
            filter_arg = "arnndn"
    elif filter_type == "afftdn":
        filter_arg = "afftdn"
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type}")

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(input_path),
        "-af", filter_arg,
        "-c:a", "pcm_s16le",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)

