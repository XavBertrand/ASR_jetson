from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.pipeline import full_pipeline as fp
from asr_jetson.pipeline.full_pipeline import PipelineConfig


@pytest.mark.integration
def test_pipeline_ner_unavailable_falls_back_with_canonical_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    transcript = tmp_path / "run" / "txt" / "sample.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("Contact: alice.martin@example.com", encoding="utf-8")

    def _raise_init(*args, **kwargs):
        raise transformer_module.TransformerBackendInitializationError("init failed")

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _raise_init)

    cfg = PipelineConfig(anonymize=True, generate_meeting_report=False, anon_enable_llm_qc=False)
    outputs, _ = fp._run_postprocessing(
        base_text_path=transcript,
        run_root=tmp_path / "run",
        cfg=cfg,
        audio_path=tmp_path / "audio.wav",
        run_id="CASE-FALLBACK",
        meeting_date="2026-02-25",
        run_time_label="120004",
    )

    anon_text = Path(outputs["txt_anon"]).read_text(encoding="utf-8")
    assert "alice.martin@example.com" not in anon_text
    assert "<EMAIL_" in anon_text

    mapping = json.loads(Path(outputs["anon_mapping"]).read_text(encoding="utf-8"))
    warnings = mapping.get("warnings", [])
    assert warnings, "fallback warning must be emitted"
    codes = {w.get("warning_code") for w in warnings}
    assert "NER_UNAVAILABLE_REGEX_FALLBACK" in codes
