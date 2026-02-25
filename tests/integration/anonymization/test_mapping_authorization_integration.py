from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_mapping_resolution_requires_internal_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    key_bytes = b"0123456789ABCDEF0123456789ABCDEF"
    monkeypatch.setenv("ANON_KEY_PROVIDER", "env")
    monkeypatch.setenv("ANON_KEY_ID", "anon-key-v1")
    monkeypatch.setenv("ANON_MAPPING_KEY", base64.b64encode(key_bytes).decode("ascii"))
    monkeypatch.setenv("ANON_INTERNAL_API_KEY", "secret-internal-key")

    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.txt")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-MAP-AUTH",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
            "--mapping",
            "always",
        ]
    )
    assert exit_code == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    doc = payload["documents"][0]
    mapping_path = doc["mapping_path"]

    denied_exit = anonymize_main(
        [
            "--resolve-mapping",
            mapping_path,
            "--case-id",
            "CASE-MAP-AUTH",
            "--resolve-document-id",
            doc["document_id"],
            "--internal-api-key",
            "wrong-key",
        ]
    )
    captured = capsys.readouterr().out
    assert denied_exit == 40
    assert "SECURITY_POLICY_ERROR" in captured
    assert "access denied" in captured.lower()
    assert "alice.martin@example.com" not in captured
