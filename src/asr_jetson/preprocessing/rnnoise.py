"""Noise reduction helpers built around ffmpeg RNNoise filters."""
import subprocess
from pathlib import Path
from typing import Optional


def apply_rnnoise(
    input_path: Path,
    output_path: Path,
    model_path: Optional[Path] = None,
    filter_type: str = "arnndn",
) -> None:
    """
    Apply noise reduction via ffmpeg using ``arnndn`` when available, or ``afftdn``.

    :param input_path: Source audio file path.
    :type input_path: Path
    :param output_path: Destination path for the denoised WAV file.
    :type output_path: Path
    :param model_path: Optional RNNoise ``.nn`` model path (required for ``arnndn``).
    :type model_path: Optional[Path]
    :param filter_type: Filter to apply; ``"arnndn"`` (default) or ``"afftdn"`` fallback.
    :type filter_type: str
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

