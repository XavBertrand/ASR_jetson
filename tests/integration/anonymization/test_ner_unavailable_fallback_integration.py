import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_ner_unavailable_degrades_with_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_ANON_DISABLE_NER", "1")
    report = tmp_path / "report.json"

    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US1-NER-OFF",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    warnings = [
        code
        for document in payload.get("documents", [])
        for code in document.get("warning_codes", [])
    ]
    assert "NER_UNAVAILABLE" in warnings

    txt_output = (tmp_path / "anonymized" / "sample.txt").read_text(encoding="utf-8")
    assert "alice.martin@example.com" not in txt_output
    assert "<EMAIL_" in txt_output
