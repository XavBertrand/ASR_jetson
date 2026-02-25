from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_batch_resilience_continues_after_single_document_failure(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    (input_dir / "ok.txt").write_text("Client: Alice Martin\nEmail: alice.martin@example.com\n", encoding="utf-8")
    pii_token = "resilience.failure.token@example.com"
    (input_dir / "broken.docx").write_text(pii_token, encoding="utf-8")

    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US3-RESILIENCE",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 10

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["totals"]["total_documents"] == 2
    assert payload["totals"]["succeeded"] >= 1
    assert payload["totals"]["failed"] >= 1

    failed_docs = [doc for doc in payload["documents"] if doc["status"] == "failed"]
    assert failed_docs
    assert all(doc["failure_code"] == "PROCESSING_ERROR" for doc in failed_docs)
    assert pii_token not in json.dumps(failed_docs)

    assert (tmp_path / "anonymized" / "ok.txt").exists()
