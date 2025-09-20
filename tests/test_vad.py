import pytest
from pathlib import Path
from preprocessing.vad import apply_vad


def test_vad(tmp_path: Path):
    """
    Test VAD stub (will be replaced with real Silero).
    """
    input_file = tmp_path / "fake.wav"
    output_file = tmp_path / "segments.json"

    # Create empty fake wav
    input_file.write_bytes(b"FAKE_WAV")

    # Run VAD
    apply_vad(input_file, output_file)

    # Assertions
    assert output_file.exists(), "Segments file not created"
    data = output_file.read_text(encoding="utf-8")
    assert "start" in data and "end" in data, "Segments missing fields"
