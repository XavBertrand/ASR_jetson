"""Typed contracts for ASR pipeline text backend integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CANONICAL_WARNING_CODE = "NER_UNAVAILABLE_REGEX_FALLBACK"
CANONICAL_WARNING_LEVEL = "WARNING"
CANONICAL_WARNING_MESSAGE = "NER unavailable => regex-only fallback"


@dataclass(frozen=True)
class BackendWarning:
    warning_code: str
    warning_level: str
    warning_message: str


@dataclass(frozen=True)
class TextAnonymizationRequest:
    text: str
    domain_entities: dict[str, list[str]] | None = None
    preserve_dates: bool = True
    model_name: str = "urchade/gliner_multi_pii-v1"
    device: int | str | None = "cuda"
    case_id: str | None = None


@dataclass
class TextAnonymizationResult:
    anonymized_text: str
    mapping: dict[str, Any]
    warnings: list[BackendWarning] = field(default_factory=list)
    mode: str = "nominal"


def canonical_fallback_warning() -> BackendWarning:
    return BackendWarning(
        warning_code=CANONICAL_WARNING_CODE,
        warning_level=CANONICAL_WARNING_LEVEL,
        warning_message=CANONICAL_WARNING_MESSAGE,
    )
