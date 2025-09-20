import logging
from pathlib import Path

def transcribe_audio(audio_path: Path, diarization_file: Path, output_path: Path) -> None:
    """
    Stub for ASR transcription using NeMo FastConformer/Parakeet/Canary.
    Currently just logs and writes dummy transcript.
    """
    logging.info("Transcribing audio %s with diarization %s -> %s", audio_path, diarization_file, output_path)
    fake_transcript = """[spk0] Hello, this is a fake transcript.
[spk1] Yes, and this is another fake reply."""
    output_path.write_text(fake_transcript, encoding="utf-8")
