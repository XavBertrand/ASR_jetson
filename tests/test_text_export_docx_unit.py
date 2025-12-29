from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from asr_jetson.postprocessing import docx_export
from asr_jetson.postprocessing.text_export import write_dialogue_txt


@pytest.mark.unit
def test_write_dialogue_txt_merges_turns(tmp_path: Path) -> None:
    segments = [
        {"start": 2.0, "end": 2.2, "text": "Oui", "speaker": 1},
        {"start": 1.0, "end": 1.2, "text": "Bonjour", "speaker": 0},
        {"start": 1.3, "end": 1.5, "text": "ca va ?", "speaker": 0},
        {"start": 2.3, "end": 2.4, "text": "merci", "speaker": 1},
    ]
    out_path = tmp_path / "dialogue.txt"
    write_dialogue_txt(segments, out_path, one_based=True)

    content = out_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    assert lines[0].startswith("SPEAKER_1 : Bonjour")
    assert "ca va?" in lines[0]
    assert lines[1].startswith("SPEAKER_2 : Oui merci")


@pytest.mark.unit
def test_docx_fallback_writer_creates_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docx_export, "_HAS_PYTHON_DOCX", False)

    out_path = tmp_path / "report.docx"
    docx_export.save_docx_from_markdown_sections("### Title\n\nBody", str(out_path), title="Report")

    assert out_path.exists()
    with ZipFile(out_path) as zf:
        assert "word/document.xml" in zf.namelist()
