from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_pdf_metadata_is_sanitized(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.pdf")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US1-META",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    text = (tmp_path / "anonymized" / "sample.pdf").read_bytes().decode("latin-1", errors="ignore")
    assert "/Author ()" in text
    assert "/Creator ()" in text
    assert "Alice Martin" not in text
