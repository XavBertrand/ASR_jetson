from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_mixed_language_and_abbreviations_redaction_with_warning(tmp_path: Path) -> None:
    input_file = tmp_path / "mixed.txt"
    input_file.write_text(
        "Bonjour Mme Dupont. Contact EN: jean.dupont@example.com. Dr J reviewed dossier AB-1234.",
        encoding="utf-8",
    )

    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(input_file),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US3-MIXED",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    doc = payload["documents"][0]
    assert "AMBIGUOUS_ENTITY" in doc["warning_codes"]

    anonymized = (tmp_path / "anonymized" / "mixed.txt").read_text(encoding="utf-8")
    assert "jean.dupont@example.com" not in anonymized
