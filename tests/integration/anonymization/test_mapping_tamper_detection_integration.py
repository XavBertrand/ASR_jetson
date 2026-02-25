from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main
from asr_jetson.anonymization.core.errors import SecurityPolicyError
from asr_jetson.anonymization.storage.mapping_store import MappingStore


@pytest.mark.integration
def test_mapping_tamper_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            "CASE-MAP-TAMPER",
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

    artifact = json.loads(mapping_path.read_text(encoding="utf-8"))
    ciphertext = artifact["ciphertext_b64"]
    artifact["ciphertext_b64"] = ("A" if ciphertext[0] != "A" else "B") + ciphertext[1:]
    mapping_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    with pytest.raises(SecurityPolicyError) as exc:
        MappingStore().read_mapping(
            case_id="CASE-MAP-TAMPER",
            document_id=doc["document_id"],
            mapping_path=mapping_path,
        )

    assert exc.value.code == "SECURITY_POLICY_ERROR"
    assert "verification failed" in exc.value.message_safe.lower()
