from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.pipeline import full_pipeline as fp
from asr_jetson.pipeline.full_pipeline import PipelineConfig


class RuntimeLimitExceededError(RuntimeError):
    """Synthetic runtime-limit exception used to validate bounded behavior propagation."""


@pytest.mark.integration
def test_runtime_limit_bounded_behavior_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    sensitive_fragment = "VERY_LONG_SENSITIVE_TRANSCRIPT_FRAGMENT"
    oversized_text = f"{sensitive_fragment} " + ("Alice Martin " * 10_000)

    transcript = tmp_path / "run" / "txt" / "sample.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(oversized_text, encoding="utf-8")

    def _raise_runtime_limit(text: str, *args, **kwargs):
        if len(text) > 5_000:
            raise RuntimeLimitExceededError(f"runtime_limit_exceeded: {sensitive_fragment}")
        return text, {}

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _raise_runtime_limit)

    cfg = PipelineConfig(anonymize=True, generate_meeting_report=False, anon_enable_llm_qc=False)
    with pytest.raises(RuntimeError) as exc_info:
        fp._run_postprocessing(
            base_text_path=transcript,
            run_root=tmp_path / "run",
            cfg=cfg,
            audio_path=tmp_path / "audio.wav",
            run_id="CASE-RUNTIME-LIMIT",
            meeting_date="2026-02-25",
            run_time_label="120006",
        )

    message = str(exc_info.value)
    assert "RuntimeLimitExceededError" in message
    assert sensitive_fragment not in message
