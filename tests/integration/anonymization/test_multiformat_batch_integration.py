import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_multiformat_batch_cli_generates_outputs(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US1-001",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    out_root = tmp_path / "anonymized"
    assert (out_root / "sample.txt").exists()
    assert (out_root / "sample.pdf").exists()
    assert (out_root / "sample.docx").exists()
    assert (out_root / "sample.xlsx").exists()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["totals"]["total_documents"] == 4
    assert payload["totals"]["failed"] == 0
