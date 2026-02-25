from pathlib import Path

import pytest

from asr_jetson.anonymization.parsers.docx_parser import DocxParser
from asr_jetson.anonymization.parsers.pdf_parser import PdfParser
from asr_jetson.anonymization.parsers.txt_parser import TxtParser
from asr_jetson.anonymization.parsers.xlsx_parser import XlsxParser
from asr_jetson.anonymization.renderers.docx_renderer import DocxRenderer
from asr_jetson.anonymization.renderers.pdf_renderer import PdfRenderer
from asr_jetson.anonymization.renderers.txt_renderer import TxtRenderer
from asr_jetson.anonymization.renderers.xlsx_renderer import XlsxRenderer


FIXTURE_DIR = Path("tests/data/anonymization/fixtures/us1")


@pytest.mark.unit
def test_txt_renderer_rewrites_sensitive_values(tmp_path: Path) -> None:
    out = tmp_path / "sample.txt"
    TxtRenderer().render(FIXTURE_DIR / "sample.txt", out, {"Alice Martin": "<PERSON_001>"})
    text = out.read_text(encoding="utf-8")
    assert "Alice Martin" not in text
    assert "<PERSON_001>" in text


@pytest.mark.unit
def test_pdf_renderer_rewrites_stream_and_sanitizes_metadata(tmp_path: Path) -> None:
    out = tmp_path / "sample.pdf"
    PdfRenderer().render(FIXTURE_DIR / "sample.pdf", out, {"Alice Martin": "<PERSON_001>"})
    parsed = PdfParser().parse(out, "x")
    assert "Alice Martin" not in parsed.text
    assert "/Author ()" in parsed.text


@pytest.mark.unit
def test_docx_renderer_rewrites_body_and_core_props(tmp_path: Path) -> None:
    out = tmp_path / "sample.docx"
    DocxRenderer().render(FIXTURE_DIR / "sample.docx", out, {"Alice Martin": "<PERSON_001>"})
    parsed = DocxParser().parse(out, "x")
    assert "Alice Martin" not in parsed.text
    assert "<PERSON_001>" in parsed.text


@pytest.mark.unit
def test_xlsx_renderer_rewrites_xml_text_nodes(tmp_path: Path) -> None:
    out = tmp_path / "sample.xlsx"
    XlsxRenderer().render(FIXTURE_DIR / "sample.xlsx", out, {"Alice Martin": "<PERSON_001>"})
    parsed = XlsxParser().parse(out, "x")
    assert "Alice Martin" not in parsed.text
    assert "<PERSON_001>" in parsed.text
