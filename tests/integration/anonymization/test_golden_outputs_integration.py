from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main
from asr_jetson.anonymization.parsers.docx_parser import DocxParser
from asr_jetson.anonymization.parsers.pdf_parser import PdfParser
from asr_jetson.anonymization.parsers.txt_parser import TxtParser
from asr_jetson.anonymization.parsers.xlsx_parser import XlsxParser


@pytest.mark.integration
def test_outputs_match_golden_expectations(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(Path("tests/data/anonymization/fixtures/us1")),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US1-GOLDEN",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0

    outputs = {
        "txt": TxtParser().parse(tmp_path / "anonymized" / "sample.txt", "txt").text,
        "pdf": PdfParser().parse(tmp_path / "anonymized" / "sample.pdf", "pdf").text,
        "docx": DocxParser().parse(tmp_path / "anonymized" / "sample.docx", "docx").text,
        "xlsx": XlsxParser().parse(tmp_path / "anonymized" / "sample.xlsx", "xlsx").text,
    }

    for fmt, text in outputs.items():
        golden = Path(f"tests/data/anonymization/golden/us1/{fmt}.txt").read_text(encoding="utf-8")
        for line in [ln.strip() for ln in golden.splitlines() if ln.strip()]:
            assert line in text
