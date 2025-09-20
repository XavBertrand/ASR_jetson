"""
Pipeline orchestration for lightweight ASR system.

Steps:
1. Denoising with RNNoise
2. Voice Activity Detection (Silero VAD)
3. Speaker diarization (TitaNet-S embeddings + spectral clustering)
4. ASR transcription (FastConformer / Canary / Parakeet)
"""

import logging
from pathlib import Path

# Import stubs for each module (we'll implement them later)
from preprocessing.rnnoise import apply_rnnoise
from preprocessing.vad import apply_vad
from diarization.titanet import extract_speaker_embeddings
from diarization.clustering import cluster_speakers
from asr.fastconformer import transcribe_audio


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ASRPipeline:
    def __init__(self, config: dict):
        """
        Initialize the pipeline with a configuration dictionary.
        """
        self.config = config

    def run(self, input_file: str, output_dir: str = "outputs") -> Path:
        """
        Run the full ASR pipeline.

        Args:
            input_file (str): Path to input audio file
            output_dir (str): Path where to save results

        Returns:
            Path: Path to the final transcript file
        """
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logging.info("Starting ASR pipeline on file: %s", input_file)

        # Step 1: Denoising
        denoised_file = output_path / "denoised.wav"
        apply_rnnoise(input_path, denoised_file)

        # Step 2: Voice Activity Detection
        segments_file = output_path / "segments.json"
        apply_vad(denoised_file, segments_file)

        # Step 3: Speaker Embeddings + Clustering
        embeddings_file = output_path / "embeddings.npy"
        extract_speaker_embeddings(denoised_file, segments_file, embeddings_file)

        diarization_file = output_path / "diarization.json"
        cluster_speakers(embeddings_file, diarization_file)

        # Step 4: ASR Transcription
        transcript_file = output_path / "transcript.txt"
        transcribe_audio(denoised_file, diarization_file, transcript_file)

        logging.info("Pipeline completed. Transcript saved at: %s", transcript_file)
        return transcript_file


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run ASR pipeline on an audio file")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml", help="Path to config file")
    args = parser.parse_args()

    # For now, config is a dummy dict; later we'll load YAML
    config = {"asr_model": "fastconformer", "sample_rate": 16000}

    pipeline = ASRPipeline(config)
    pipeline.run(args.input_file)
