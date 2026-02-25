"""Regex-based sensitive entity detector."""

from __future__ import annotations

import re

from asr_jetson.anonymization.core.models import Entity


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?:\+?\d[\d .\-]{7,}\d)")
_ACCOUNT_RE = re.compile(r"\b\d{8,}\b")


class RegexDetector:
    def detect(self, text: str, document_id: str) -> list[Entity]:
        entities: list[Entity] = []
        idx = 0
        for entity_type, pattern in (
            ("email", _EMAIL_RE),
            ("phone", _PHONE_RE),
            ("account", _ACCOUNT_RE),
        ):
            for match in pattern.finditer(text):
                idx += 1
                entities.append(
                    Entity(
                        entity_id=f"re:{idx}",
                        document_id=document_id,
                        span_id=f"{document_id}:{match.start()}:{match.end()}",
                        entity_type=entity_type,
                        value=match.group(0),
                        detector_source="regex",
                        confidence=1.0,
                    )
                )
        return entities
