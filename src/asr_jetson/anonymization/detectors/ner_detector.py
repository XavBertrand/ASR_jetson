"""Local NER detector wrapper with explicit unavailable mode."""

from __future__ import annotations

import os
import re

from asr_jetson.anonymization.core.models import Entity


class NERUnavailableError(RuntimeError):
    """Raised when NER is intentionally unavailable."""


_NAME_RE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")


class NerDetector:
    def detect(self, text: str, document_id: str) -> list[Entity]:
        if os.environ.get("ASR_ANON_DISABLE_NER", "").strip().lower() in {"1", "true", "yes"}:
            raise NERUnavailableError("NER model unavailable")

        entities: list[Entity] = []
        for idx, match in enumerate(_NAME_RE.finditer(text), start=1):
            entities.append(
                Entity(
                    entity_id=f"ner:{idx}",
                    document_id=document_id,
                    span_id=f"{document_id}:{match.start()}:{match.end()}",
                    entity_type="person",
                    value=match.group(0),
                    detector_source="ner",
                    confidence=0.8,
                )
            )
        return entities
