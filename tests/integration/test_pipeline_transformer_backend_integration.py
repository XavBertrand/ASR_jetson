from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.pipeline import full_pipeline as fp
from asr_jetson.pipeline.full_pipeline import PipelineConfig


@pytest.mark.integration
def test_pipeline_integration_uses_canonical_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    transcript = tmp_path / "run" / "txt" / "sample.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("Alice Martin email alice.martin@example.com", encoding="utf-8")

    call_count = {"count": 0}

    def _fake_run(*args, **kwargs):
        call_count["count"] += 1
        return (
            "<PERSON_001> <EMAIL_001>",
            {
                "entities": {
                    "<PERSON_001>": {"label": "PERSON", "canonical": "Alice Martin", "values": ["Alice Martin"]},
                    "<EMAIL_001>": {"label": "EMAIL", "canonical": "alice.martin@example.com", "values": ["alice.martin@example.com"]},
                },
                "reverse_map": {
                    "<PERSON_001>": "Alice Martin",
                    "<EMAIL_001>": "alice.martin@example.com",
                },
                "pseudonym_map": {
                    "<PERSON_001>": "<PERSON_001>",
                    "<EMAIL_001>": "<EMAIL_001>",
                },
                "pseudonym_reverse_map": {
                    "<PERSON_001>": "Alice Martin",
                    "<EMAIL_001>": "alice.martin@example.com",
                },
                "stats": {"total": 2, "by_type": {"PERSON": 1, "EMAIL": 1}},
            },
        )

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _fake_run)

    cfg = PipelineConfig(anonymize=True, generate_meeting_report=False, anon_enable_llm_qc=False)
    outputs, _ = fp._run_postprocessing(
        base_text_path=transcript,
        run_root=tmp_path / "run",
        cfg=cfg,
        audio_path=tmp_path / "audio.wav",
        run_id="CASE-INT-001",
        meeting_date="2026-02-25",
        run_time_label="120001",
    )

    assert call_count["count"] == 1
    anon_text = Path(outputs["txt_anon"]).read_text(encoding="utf-8")
    assert "<PERSON_" in anon_text
    assert "<EMAIL_" in anon_text
    payload = json.loads(Path(outputs["anon_mapping"]).read_text(encoding="utf-8"))
    assert "reverse_map" in payload
