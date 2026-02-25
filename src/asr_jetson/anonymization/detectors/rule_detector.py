"""Rule-based detector for structured business markers."""

from __future__ import annotations

import re

from asr_jetson.anonymization.core.models import Entity


_CASE_RE = re.compile(r"\bCASE-[A-Z0-9\-]+\b")


class RuleDetector:
    def detect(self, text: str, document_id: str) -> list[Entity]:
        entities: list[Entity] = []
        for idx, match in enumerate(_CASE_RE.finditer(text), start=1):
            entities.append(
                Entity(
                    entity_id=f"rule:{idx}",
                    document_id=document_id,
                    span_id=f"{document_id}:{match.start()}:{match.end()}",
                    entity_type="case_reference",
                    value=match.group(0),
                    detector_source="rules",
                    confidence=0.95,
                )
            )
        return entities
