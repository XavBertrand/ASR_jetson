"""XLSX parser reading visible and hidden workbook XML text nodes."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
import zipfile

from asr_jetson.anonymization.core.models import ParsedDocument, Span


class XlsxParser:
    _NS = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    def _iter_strings(self, xml_payload: str) -> list[str]:
        root = ET.fromstring(xml_payload)
        strings: list[str] = []
        for node in root.findall(".//m:t", self._NS):
            if node.text:
                strings.append(node.text)
        for node in root.findall(".//dc:title", self._NS):
            if node.text:
                strings.append(node.text)
        return strings

    def parse(self, input_path: Path, document_id: str) -> ParsedDocument:
        parts: list[str] = []
        with zipfile.ZipFile(input_path, "r") as archive:
            for name in archive.namelist():
                if not name.endswith(".xml"):
                    continue
                if not (
                    name.startswith("xl/")
                    or name.startswith("docProps/")
                ):
                    continue
                payload = archive.read(name).decode("utf-8", errors="ignore")
                parts.extend(self._iter_strings(payload))

        text = "\n".join(parts)
        span = Span(
            span_id=f"{document_id}:0",
            document_id=document_id,
            start=0,
            end=len(text),
            anchor_type="xlsx_cell",
            anchor_ref="xl/*",
        )
        return ParsedDocument(document_id=document_id, format="xlsx", text=text, spans=[span])
