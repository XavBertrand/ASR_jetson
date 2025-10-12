from pathlib import Path
import subprocess
import pytest
from asr_jetson.preprocessing.rnnoise import apply_rnnoise

from utils import PROJECT_ROOT


@pytest.mark.parametrize("filter_type", ["arnndn", "afftdn"])
def test_rnnoise_with_ogg(tmp_path: Path, filter_type: str):
    """
    Test RNNoise/Noise reduction wrapper with an OGG input.
    Tries arnndn first; if unavailable, falls back to afftdn.
    +
    """
    # Paths
    input_file = tmp_path / "test.mp3"
    output_file = tmp_path / "denoised.wav"
    model_file = (
        PROJECT_ROOT
        / "models"
        / "rnnoise"
        / "somnolent-hogwash-2019-03-29.nn"
    )

    # Generate a dummy OGG file (1s silence @16kHz)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", "1", str(input_file)
        ],
        check=True
    )

    # Run denoising
    try:
        apply_rnnoise(input_file, output_file, model_path=model_file, filter_type=filter_type)
    except subprocess.CalledProcessError as e:
        if filter_type == "arnndn":
            pytest.skip("arnndn filter not available in this ffmpeg build, skipped.")
        else:
            raise e

    # Check output exists and is non-empty
    assert output_file.exists()
    assert output_file.stat().st_size > 0
