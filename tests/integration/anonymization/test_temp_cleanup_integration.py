from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_temp_workspace_cleanup_on_success(tmp_path: Path) -> None:
    report = tmp_path / "report-success.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.txt")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US3-CLEANUP-SUCCESS",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    tmp_root = tmp_path / "intermediate" / "tmp"
    leftovers = list(tmp_root.glob("anon_*")) if tmp_root.exists() else []
    assert leftovers == []


@pytest.mark.integration
def test_temp_workspace_cleanup_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASR_ANON_FORCE_CLEANUP_FAILURE", "1")

    report = tmp_path / "report-failure.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.txt")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US3-CLEANUP-FAIL",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["documents"]
    assert "CLEANUP_FAILED" in payload["documents"][0]["warning_codes"]

    audit_file = tmp_path / "audit" / "CASE-US3-CLEANUP-FAIL.jsonl"
    assert audit_file.exists()
    audit_lines = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines() if line]
    assert any(line["event_code"] == "cleanup_failed" for line in audit_lines)

    tmp_root = tmp_path / "intermediate" / "tmp"
    leftovers = list(tmp_root.glob("anon_*")) if tmp_root.exists() else []
    assert leftovers
