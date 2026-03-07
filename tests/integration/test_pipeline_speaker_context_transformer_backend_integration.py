from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.pipeline import full_pipeline as fp
from asr_jetson.pipeline.full_pipeline import PipelineConfig


@pytest.mark.integration
def test_speaker_context_also_uses_canonical_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    transcript = tmp_path / "run" / "txt" / "sample.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("Intervention de Alice Martin", encoding="utf-8")

    calls: list[str] = []

    def _fake_run(text: str, *args, **kwargs):
        calls.append(text)
        token = "<PERSON_001>"
        return (
            token,
            {
                "entities": {token: {"label": "PERSON", "canonical": text, "values": [text]}},
                "reverse_map": {token: text},
                "pseudonym_map": {token: token},
                "pseudonym_reverse_map": {token: text},
                "stats": {"total": 1, "by_type": {"PERSON": 1}},
            },
        )

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _fake_run)

    cfg = PipelineConfig(
        anonymize=True,
        generate_meeting_report=False,
        anon_enable_llm_qc=False,
        speaker_context="Alice Martin est l'avocate",
    )
    outputs, _ = fp._run_postprocessing(
        base_text_path=transcript,
        run_root=tmp_path / "run",
        cfg=cfg,
        audio_path=tmp_path / "audio.wav",
        run_id="CASE-INT-CTX",
        meeting_date="2026-02-25",
        run_time_label="120002",
    )

    assert len(calls) == 2
    payload = json.loads(Path(outputs["anon_mapping"]).read_text(encoding="utf-8"))
    assert payload["stats"]["total"] >= 1
