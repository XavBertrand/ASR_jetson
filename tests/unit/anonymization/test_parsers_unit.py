from pathlib import Path

import pytest

from asr_jetson.anonymization.parsers.docx_parser import DocxParser
from asr_jetson.anonymization.parsers.pdf_parser import PdfParser
from asr_jetson.anonymization.parsers.txt_parser import TxtParser
from asr_jetson.anonymization.parsers.xlsx_parser import XlsxParser


FIXTURE_DIR = Path("tests/data/anonymization/fixtures/us1")


@pytest.mark.unit
def test_txt_parser_extracts_text_and_span() -> None:
    parsed = TxtParser().parse(FIXTURE_DIR / "sample.txt", "doc1")
    assert "Alice Martin" in parsed.text
    assert parsed.spans and parsed.spans[0].end > parsed.spans[0].start


@pytest.mark.unit
def test_pdf_parser_extracts_text_and_span() -> None:
    parsed = PdfParser().parse(FIXTURE_DIR / "sample.pdf", "doc2")
    assert "alice.martin@example.com" in parsed.text
    assert parsed.spans and parsed.spans[0].anchor_type == "pdf_quad"


@pytest.mark.unit
def test_docx_parser_extracts_body_and_metadata() -> None:
    parsed = DocxParser().parse(FIXTURE_DIR / "sample.docx", "doc3")
    assert "Alice Martin" in parsed.text
    assert "Confidential note" in parsed.text


@pytest.mark.unit
def test_xlsx_parser_extracts_sheet_text() -> None:
    parsed = XlsxParser().parse(FIXTURE_DIR / "sample.xlsx", "doc4")
    assert "Alice Martin" in parsed.text
    assert "CASE-ALPHA-01" in parsed.text
