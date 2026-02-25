from __future__ import annotations

import base64
import json
import stat
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main
from asr_jetson.anonymization.storage.mapping_store import MappingStore


@pytest.mark.integration
def test_mapping_artifact_is_encrypted_and_recoverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_bytes = b"0123456789ABCDEF0123456789ABCDEF"
    monkeypatch.setenv("ANON_KEY_PROVIDER", "env")
    monkeypatch.setenv("ANON_KEY_ID", "anon-key-v1")
    monkeypatch.setenv("ANON_MAPPING_KEY", base64.b64encode(key_bytes).decode("ascii"))

    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.txt")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-MAP-ENC",
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
    mapping_path = Path(doc["mapping_path"])
    assert mapping_path.exists()

    artifact_text = mapping_path.read_text(encoding="utf-8")
    assert "alice.martin@example.com" not in artifact_text
    assert stat.S_IMODE(mapping_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(mapping_path.parent.stat().st_mode) == 0o700

    mapping = MappingStore().read_mapping(
        case_id="CASE-MAP-ENC",
        document_id=doc["document_id"],
        mapping_path=mapping_path,
    )
    assert "alice.martin@example.com" in mapping
    assert mapping["alice.martin@example.com"].startswith("<EMAIL_")
