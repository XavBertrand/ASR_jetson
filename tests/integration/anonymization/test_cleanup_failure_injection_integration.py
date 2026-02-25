from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_cleanup_failure_injection_emits_warning_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASR_ANON_FORCE_CLEANUP_FAILURE", "true")

    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.txt")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US3-INJECT",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert "CLEANUP_FAILED" in payload["documents"][0]["warning_codes"]

    audit_path = tmp_path / "audit" / "CASE-US3-INJECT.jsonl"
    assert audit_path.exists()
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    cleanup_events = [event for event in events if event.get("event_code") == "cleanup_failed"]
    assert cleanup_events
    assert "alice.martin@example.com" not in json.dumps(cleanup_events)
