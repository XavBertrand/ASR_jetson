"""XLSX renderer anonymizing workbook XML text values."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
import zipfile


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
    for source in sorted(replacements.keys(), key=len, reverse=True):
        text = text.replace(source, replacements[source])
    return text


class XlsxRenderer:
    _TEXT_TAGS = {
        "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t",
        "{http://purl.org/dc/elements/1.1/}title",
        "{http://purl.org/dc/elements/1.1/}creator",
        "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}keywords",
    }

    def _rewrite_xml(self, payload: bytes, replacements: dict[str, str]) -> bytes:
        try:
            root = ET.fromstring(payload.decode("utf-8", errors="ignore"))
        except ET.ParseError:
            return payload

        changed = False
        for node in root.iter():
            if node.tag in self._TEXT_TAGS and node.text:
                new_text = _apply_replacements(node.text, replacements)
                if new_text != node.text:
                    node.text = new_text
                    changed = True

        if not changed:
            return payload
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def render(self, input_path: Path, output_path: Path, replacements: dict[str, str]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as src, zipfile.ZipFile(output_path, "w") as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename.endswith(".xml") and (
                    item.filename.startswith("xl/") or item.filename.startswith("docProps/")
                ):
                    data = self._rewrite_xml(data, replacements)
                dst.writestr(item, data)
