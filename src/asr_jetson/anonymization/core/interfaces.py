"""Protocol interfaces for anonymization layers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from asr_jetson.anonymization.core.models import Entity, ParsedDocument


class DocumentParser(Protocol):
    def parse(self, input_path: Path, document_id: str) -> ParsedDocument:
        """Extract text plus positional anchors from an input document."""


class EntityDetector(Protocol):
    def detect(self, text: str, document_id: str) -> list[Entity]:
        """Detect candidate sensitive entities from text."""


class DocumentRenderer(Protocol):
    def render(self, input_path: Path, output_path: Path, replacements: dict[str, str]) -> None:
        """Render anonymized output with replacements applied."""
