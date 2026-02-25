import os
from pathlib import Path

import pytest

from asr_jetson.anonymization.core.models import DocumentRequest
from asr_jetson.anonymization.core.policy import load_policy
from asr_jetson.anonymization.core.service import DocumentAnonymizer


@pytest.mark.unit
def test_ner_unavailable_falls_back_to_regex_with_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_ANON_DISABLE_NER", "1")

    policy = load_policy("strict_offline", Path("configs/anonymization_profiles.yaml"))
    service = DocumentAnonymizer()
    in_file = Path("tests/data/anonymization/fixtures/us1/sample.txt")
    out_file = tmp_path / "sample.txt"

    result = service.anonymize_document(
        DocumentRequest(
            case_id="CASE-UNIT-1",
            policy=policy,
            input_path=in_file,
            output_path=out_file,
            format_hint="txt",
        )
    )

    output = out_file.read_text(encoding="utf-8")
    assert result.status == "degraded"
    assert "NER_UNAVAILABLE" in result.warning_codes
    assert "alice.martin@example.com" not in output
    assert "<EMAIL_" in output

    monkeypatch.delenv("ASR_ANON_DISABLE_NER", raising=False)
