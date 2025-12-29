from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.utils.config import AppCfg, load_config


@pytest.mark.unit
def test_load_config_default() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, AppCfg)
    assert cfg.device == "cuda"


@pytest.mark.unit
def test_load_config_from_yaml(tmp_path: Path) -> None:
    content = """
runtime:
  device: cpu
  batch_size: 2
  vad: silero
  asr: fasterwhisper
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(content, encoding="utf-8")

    cfg = load_config(str(cfg_path))
    assert cfg.device == "cpu"
    assert cfg.batch_size == 2
    assert cfg.vad == "silero"
