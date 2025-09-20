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

# Import modules
from preprocessing.rnnoise import apply_rnnoise
from preprocessing.vad import apply_vad
from diarization.titanet import extract_speaker_embeddings
from diarization.clustering import cluster_speakers
from asr.fastconformer import transcribe_audio


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ModelPaths:
    """
    Central place to manage paths to pre-trained models.
    """

    def __init__(self, root: Path):
        self.root = root

        # RNNoise model
        self.rnnoise = root / "rnnoise" / "somnolent-hogwash-2019-03-29.nn"

        # Silero (PyTorch .pt or TorchHub)
        self.silero = root / "silero"

        # NeMo models (downloaded later if not present)
        self.nemo = root / "nemo"

    def check(self):
        """
        Log missing models (does not stop execution yet).
        """
        if not self.rnnoise.exists():
            logging.warning("RNNoise model not found at %s", self.rnnoise)


class ASRPipeline:
    def __init__(self, config: dict, models: ModelPaths):
        """
        Initialize the pipeline with config and model paths.
        """
        self.config = config
        self.models = models

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

        # Step 1: Denoising with RNNoise
        denoised_file = output_path / "denoised.wav"
        apply_rnnoise(input_path, denoised_file, model_path=self.models.rnnoise)

        # Step 2: VAD
        segments_file = output_path / "segments.json"
        apply_vad(denoised_file, segments_file)

        # Step 3: Speaker embeddings + clustering
        embeddings_file = output_path / "embeddings.npy"
        extract_speaker_embeddings(denoised_file, segments_file, embeddings_file)

        diarization_file = output_path / "diarization.json"
        cluster_speakers(embeddings_file, diarization_file)

        # Step 4: ASR
        transcript_file = output_path / "transcript.txt"
        transcribe_audio(denoised_file, diarization_file, transcript_file)

        logging.info("Pipeline completed. Transcript saved at: %s", transcript_file)
        return transcript_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ASR pipeline on an audio file")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml", help="Path to config file")
    args = parser.parse_args()

    # Dummy config (to be expanded later)
    config = {"asr_model": "fastconformer", "sample_rate": 16000}

    # Models
    models = ModelPaths(Path("models"))
    models.check()

    pipeline = ASRPipeline(config, models)
    pipeline.run(args.input_file)
