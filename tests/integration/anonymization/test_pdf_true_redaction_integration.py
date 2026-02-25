from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_pdf_output_does_not_contain_original_text(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1/sample.pdf")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US1-PDF",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    out_pdf = tmp_path / "anonymized" / "sample.pdf"
    data = out_pdf.read_bytes().decode("latin-1", errors="ignore")
    assert "Alice Martin" not in data
    assert "alice.martin@example.com" not in data
    assert "+33 06 11 22 33 44" not in data
    assert "<PERSON_" in data
