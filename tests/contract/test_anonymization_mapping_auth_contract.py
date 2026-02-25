from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.api.internal_routes import InternalJobStore
from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_mapping_resolution_requires_internal_api_key_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_bytes = b"0123456789ABCDEF0123456789ABCDEF"
    monkeypatch.setenv("ANON_KEY_PROVIDER", "env")
    monkeypatch.setenv("ANON_KEY_ID", "anon-key-v1")
    monkeypatch.setenv("ANON_MAPPING_KEY", base64.b64encode(key_bytes).decode("ascii"))
    monkeypatch.setenv("ANON_INTERNAL_API_KEY", "contract-secret")

    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.txt")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US3-AUTH",
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

    store = InternalJobStore()

    denied_status, denied_body = store.resolve_mapping(
        case_id="CASE-US3-AUTH",
        document_id=doc["document_id"],
        mapping_path=doc["mapping_path"],
        internal_api_key=None,
    )
    assert denied_status == 403
    assert denied_body["code"] == "FORBIDDEN"

    wrong_status, wrong_body = store.resolve_mapping(
        case_id="CASE-US3-AUTH",
        document_id=doc["document_id"],
        mapping_path=doc["mapping_path"],
        internal_api_key="wrong",
    )
    assert wrong_status == 403
    assert wrong_body["code"] == "FORBIDDEN"

    ok_status, ok_body = store.resolve_mapping(
        case_id="CASE-US3-AUTH",
        document_id=doc["document_id"],
        mapping_path=doc["mapping_path"],
        internal_api_key="contract-secret",
    )
    assert ok_status == 200
    assert ok_body["status"] == "resolved"
    assert "mapping" in ok_body
