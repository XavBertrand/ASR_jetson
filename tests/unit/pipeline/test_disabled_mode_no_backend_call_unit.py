from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.pipeline import full_pipeline as fp
from asr_jetson.pipeline.full_pipeline import PipelineConfig


@pytest.mark.unit
def test_disabled_mode_does_not_call_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    transcript = tmp_path / "run" / "txt" / "sample.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("No anonymization expected", encoding="utf-8")

    def _forbidden_backend(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("backend must not be called when anonymize=False")

    monkeypatch.setattr(fp, "_anonymize_text_via_backend", _forbidden_backend)

    cfg = PipelineConfig(anonymize=False, generate_meeting_report=False, anon_enable_llm_qc=False)
    outputs, _ = fp._run_postprocessing(
        base_text_path=transcript,
        run_root=tmp_path / "run",
        cfg=cfg,
        audio_path=tmp_path / "audio.wav",
        run_id="CASE-DISABLED",
        meeting_date="2026-02-25",
        run_time_label="120003",
    )

    assert outputs["txt_anon"] is None
    assert Path(outputs["txt"]).read_text(encoding="utf-8") == "No anonymization expected"
