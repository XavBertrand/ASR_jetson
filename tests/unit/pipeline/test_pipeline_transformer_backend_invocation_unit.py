from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.pipeline import full_pipeline as fp
from asr_jetson.pipeline.full_pipeline import PipelineConfig
from asr_jetson.pipeline.text_backend_contract import TextAnonymizationResult


@pytest.mark.unit
def test_pipeline_transcript_flow_calls_canonical_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = tmp_path / "run" / "txt" / "sample.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("Alice Martin contact alice.martin@example.com", encoding="utf-8")

    calls: list[str] = []

    def _fake_backend(*, text: str, domain_entities, cfg, case_id: str) -> TextAnonymizationResult:
        calls.append(text)
        mapping = {
            "entities": {},
            "reverse_map": {},
            "pseudonym_map": {},
            "pseudonym_reverse_map": {},
            "stats": {"total": 0, "by_type": {}},
            "corrected_text": text,
        }
        return TextAnonymizationResult(anonymized_text="<PERSON_ABC>", mapping=mapping)

    monkeypatch.setattr(fp, "_anonymize_text_via_backend", _fake_backend)

    cfg = PipelineConfig(anonymize=True, generate_meeting_report=False, anon_enable_llm_qc=False)
    outputs, _report_outputs = fp._run_postprocessing(
        base_text_path=transcript,
        run_root=tmp_path / "run",
        cfg=cfg,
        audio_path=tmp_path / "audio.wav",
        run_id="CASE-UNIT",
        meeting_date="2026-02-25",
        run_time_label="120000",
    )

    assert calls == ["Alice Martin contact alice.martin@example.com"]
    anon_text = Path(outputs["txt_anon"]).read_text(encoding="utf-8")
    assert "<PERSON_ABC>" in anon_text
    payload = json.loads(Path(outputs["anon_mapping"]).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
