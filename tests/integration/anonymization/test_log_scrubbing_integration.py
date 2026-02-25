from __future__ import annotations

import logging
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_logs_do_not_expose_fixture_pii(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pii_token = "alice.martin@example.com"

    logger_name = "asr_jetson.anonymization"
    caplog.set_level(logging.INFO, logger=logger_name)

    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.txt")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US3-LOG-SCRUB",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    stdio = capsys.readouterr()
    combined = "\n".join([caplog.text, stdio.out, stdio.err])
    assert pii_token not in combined
