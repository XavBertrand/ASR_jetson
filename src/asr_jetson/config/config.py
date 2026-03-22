"""
Centralized Pydantic configuration models for the ASR pipeline.
"""
from pathlib import Path
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, field_validator
import yaml


class DenoiseConfig(BaseModel):
    """RNNoise denoising configuration."""
    enabled: bool = True
    model_path: Path = Field(default=Path("models/rnnoise/rnnoise.rnnn"))


class VADConfig(BaseModel):
    """Silero VAD configuration."""
    model_name: str = "silero_vad"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(250, ge=100)
    min_silence_duration_ms: int = Field(100, ge=50)
    speech_pad_ms: int = Field(30, ge=0)
    sampling_rate: int = 16000


class DiarizationConfig(BaseModel):
    """Speaker diarization configuration."""
    enabled: bool = True
    pipeline: str = "pyannote/speaker-diarization-3.1"
    auth_token: Optional[str] = None
    n_speakers: Optional[int] = Field(None, ge=1, le=20)


class ASRConfig(BaseModel):
    """Automatic speech recognition engine configuration."""
    engine: Literal["faster-whisper", "fastconformer"] = "faster-whisper"
    model_size: Literal["tiny", "base", "small", "medium", "large"] = "small"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    compute_type: Literal["int8", "float16", "float32"] = "int8"
    language: Optional[str] = None
    beam_size: int = Field(5, ge=1, le=10)
    vad_filter: bool = True


class OutputConfig(BaseModel):
    """Configuration sorties"""
    output_dir: Path = Field(default=Path("outputs"))
    formats: List[Literal["json", "srt", "txt", "vtt"]] = ["json", "srt"]
    include_timestamps: bool = True
    include_confidence: bool = True

    @field_validator('output_dir')
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


class PerformanceConfig(BaseModel):
    """Configuration performance"""
    num_workers: int = Field(4, ge=1, le=16)
    batch_size: int = Field(1, ge=1, le=32)
    use_tensorrt: bool = False
    fp16: bool = True


class LoggingConfig(BaseModel):
    """Configuration logging"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Field(default=Path("logs"))
    json_logs: bool = False
    rotation: str = "500 MB"
    retention: str = "10 days"


class PipelineConfig(BaseModel):
    """Aggregate configuration for the full ASR pipeline."""
    name: str = "asr_pipeline"
    version: str = "0.1.0"

    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """
        Load a pipeline configuration from a YAML file.

        :param path: Path to the YAML configuration file.
        :type path: Path
        :returns: Instantiated pipeline configuration.
        :rtype: PipelineConfig
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """
        Serialize the pipeline configuration to a YAML file.

        :param path: Destination path for the YAML output.
        :type path: Path
        """
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    @classmethod
    def for_jetson(cls) -> "PipelineConfig":
        """
        Build a configuration tuned for NVIDIA Jetson targets.

        :returns: Jetson-optimized pipeline configuration.
        :rtype: PipelineConfig
        """
        return cls(
            asr=ASRConfig(model_size="small", device="cuda", compute_type="int8"),
            performance=PerformanceConfig(batch_size=1, num_workers=2),
            diarization=DiarizationConfig(),
        )

    @classmethod
    def for_desktop(cls) -> "PipelineConfig":
        """
        Build a configuration tuned for desktop GPUs.

        :returns: Desktop-optimized pipeline configuration.
        :rtype: PipelineConfig
        """
        return cls(
            asr=ASRConfig(model_size="medium", device="cuda", compute_type="float16"),
            performance=PerformanceConfig(batch_size=8, num_workers=8),
            diarization=DiarizationConfig(),
        )


# Create the default configuration files when executed directly.
if __name__ == "__main__":
    # Create the configuration directory.
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Default configuration.
    PipelineConfig().to_yaml(config_dir / "default.yaml")

    # Jetson configuration.
    PipelineConfig.for_jetson().to_yaml(config_dir / "jetson.yaml")

    # Desktop configuration.
    PipelineConfig.for_desktop().to_yaml(config_dir / "desktop.yaml")

    print("✓ Configuration files created in config/")
