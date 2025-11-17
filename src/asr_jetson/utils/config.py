"""
Utilities for loading lightweight application configuration blocks.
"""
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class AppCfg:
    """Configuration structure covering runtime behaviour."""

    device: str = "cuda"
    batch_size: int = 1
    vad: str = "silero"
    asr: str = "fasterwhisper"


def load_config(path: Optional[str]) -> AppCfg:
    """
    Load a configuration file and coerce it into an :class:`AppCfg` instance.

    :param path: Optional path to a YAML configuration file.
    :type path: Optional[str]
    :returns: Parsed configuration with defaults filled in when needed.
    :rtype: AppCfg
    """
    if not path:
        return AppCfg()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppCfg(**data.get("runtime", {}))
