import pytest
import os

from src.diarization.pipeline import apply_diarization

from tests.conftest import PROJECT_ROOT

def test_diarization_on_real_file():

    input_file = os.path.join(PROJECT_ROOT, "tests", "data", "test.mp3")

    assert os.path.exists(input_file), "Missing tests/data/test.mp3"

    try:
        diarized = apply_diarization(input_file, n_speakers=2, device="cpu")
    except FileNotFoundError as e:
        pytest.skip(f"TitaNet not available for tests: {e}")

    assert isinstance(diarized, list)
    assert len(diarized) > 0
