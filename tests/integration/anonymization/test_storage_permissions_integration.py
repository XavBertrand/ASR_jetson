from __future__ import annotations

import base64
import json
import stat
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_storage_permissions_are_hardened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            "CASE-US3-PERMS",
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
    mapping_path = Path(payload["documents"][0]["mapping_path"])

    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "anonymized").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "mappings").stat().st_mode) == 0o700
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert stat.S_IMODE(mapping_path.stat().st_mode) == 0o600
