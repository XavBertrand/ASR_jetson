"""TXT parser extracting plain text and byte-offset spans."""

from __future__ import annotations

from pathlib import Path

from asr_jetson.anonymization.core.streaming import read_text_streaming
from asr_jetson.anonymization.core.models import ParsedDocument, Span


class TxtParser:
    def parse(self, input_path: Path, document_id: str) -> ParsedDocument:
        text = read_text_streaming(input_path)
        span = Span(
            span_id=f"{document_id}:0",
            document_id=document_id,
            start=0,
            end=len(text),
            anchor_type="txt_offset",
            anchor_ref="0",
        )
        return ParsedDocument(document_id=document_id, format="txt", text=text, spans=[span])
