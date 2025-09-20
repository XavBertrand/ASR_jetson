from pathlib import Path
from pipeline import ASRPipeline, ModelPaths


def test_pipeline(tmp_path: Path):
    """
    Run the full pipeline (with stubs where not implemented).
    """
    input_file = tmp_path / "test.wav"

    # Generate a dummy wav (1s silence)
    import subprocess
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "1", str(input_file)
    ], check=True)

    # Init pipeline
    config = {"asr_model": "fastconformer", "sample_rate": 16000}
    models = ModelPaths(Path("models"))
    pipeline = ASRPipeline(config, models)

    # Run
    transcript_path = pipeline.run(str(input_file), output_dir=tmp_path)

    # Assertions
    assert transcript_path.exists(), "Transcript file not created"
    assert "fake" in transcript_path.read_text().lower()
