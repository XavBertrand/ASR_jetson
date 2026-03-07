from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.pipeline import full_pipeline as fp
from asr_jetson.pipeline.full_pipeline import PipelineConfig


@pytest.mark.integration
def test_runtime_exception_hard_fail_is_sanitized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    sensitive_fragment = "TOP_SENSITIVE_TRANSCRIPT_FRAGMENT_DO_NOT_EXPOSE"
    transcript = tmp_path / "run" / "txt" / "sample.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(f"Speaker: {sensitive_fragment}", encoding="utf-8")

    def _raise_runtime(*args, **kwargs):
        raise RuntimeError(f"backend execution crashed on {sensitive_fragment}")

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _raise_runtime)

    cfg = PipelineConfig(anonymize=True, generate_meeting_report=False, anon_enable_llm_qc=False)
    with pytest.raises(RuntimeError) as exc_info:
        fp._run_postprocessing(
            base_text_path=transcript,
            run_root=tmp_path / "run",
            cfg=cfg,
            audio_path=tmp_path / "audio.wav",
            run_id="CASE-RUNTIME-ERR",
            meeting_date="2026-02-25",
            run_time_label="120005",
        )

    message = str(exc_info.value)
    assert "Text anonymization failed during backend execution" in message
    assert sensitive_fragment not in message
    assert "regex-only fallback" not in message.lower()
