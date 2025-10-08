"""ASR Jetson Package"""
__version__ = "0.1.0"
__author__ = "Xavier Bertrand"

from src.pipeline.full_pipeline import run_pipeline, PipelineConfig

__all__ = ["run_pipeline", "PipelineConfig", "__version__"]
