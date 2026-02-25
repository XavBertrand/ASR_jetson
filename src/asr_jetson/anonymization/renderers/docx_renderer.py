"""DOCX renderer with body and metadata anonymization."""

from __future__ import annotations

from pathlib import Path

from docx import Document


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
    for source in sorted(replacements.keys(), key=len, reverse=True):
        text = text.replace(source, replacements[source])
    return text


class DocxRenderer:
    def render(self, input_path: Path, output_path: Path, replacements: dict[str, str]) -> None:
        doc = Document(str(input_path))
        for paragraph in doc.paragraphs:
            if paragraph.text:
                paragraph.text = _apply_replacements(paragraph.text, replacements)

        core = doc.core_properties
        for attr in ("author", "title", "subject", "comments", "category"):
            value = getattr(core, attr, None)
            if value:
                setattr(core, attr, _apply_replacements(str(value), replacements))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
