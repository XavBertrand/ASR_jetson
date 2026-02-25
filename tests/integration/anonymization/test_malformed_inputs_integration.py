from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_malformed_documents_fail_safely_and_batch_continues(tmp_path: Path) -> None:
    input_dir = tmp_path / "malformed"
    input_dir.mkdir(parents=True, exist_ok=True)

    (input_dir / "good.txt").write_text("Email: alice.martin@example.com", encoding="utf-8")
    (input_dir / "bad.docx").write_bytes(b"not-a-valid-docx")
    (input_dir / "bad.xlsx").write_bytes(b"not-a-valid-xlsx")
    (input_dir / "bad.pdf").write_bytes(b"%PDF-1.4\n\xff\xff\x00corrupted")

    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US3-MALFORMED",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 10

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["totals"]["total_documents"] == 4
    assert payload["totals"]["failed"] >= 2
    assert (tmp_path / "anonymized" / "good.txt").exists()

    failed_docs = [doc for doc in payload["documents"] if doc["status"] == "failed"]
    assert failed_docs
    for failed in failed_docs:
        assert failed["failure_code"] == "PROCESSING_ERROR"
        message = failed.get("failure_message_safe") or ""
        assert "alice.martin@example.com" not in message
