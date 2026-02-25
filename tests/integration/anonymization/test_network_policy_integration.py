from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_network_policy_default_offline_and_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ASR_ANON_REQUIRE_NETWORK", "1")

    offline_report = tmp_path / "offline-report.json"
    offline_exit = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.txt")),
            "--output",
            str(tmp_path / "offline"),
            "--case-id",
            "CASE-US3-OFFLINE",
            "--policy",
            "strict_offline",
            "--report",
            str(offline_report),
            "--mapping",
            "never",
        ]
    )
    offline_logs = capsys.readouterr().out
    assert offline_exit == 40
    assert "SECURITY_POLICY_ERROR" in offline_logs

    online_report = tmp_path / "online-report.json"
    online_exit = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.txt")),
            "--output",
            str(tmp_path / "online"),
            "--case-id",
            "CASE-US3-ONLINE",
            "--policy",
            "online_opt_in",
            "--report",
            str(online_report),
            "--mapping",
            "never",
        ]
    )
    assert online_exit == 0
    assert online_report.exists()
