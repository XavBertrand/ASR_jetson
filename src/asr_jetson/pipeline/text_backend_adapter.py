"""Canonical adapter between pipeline and transformer text anonymization backend."""

from __future__ import annotations

import hmac
import re
from hashlib import sha256
from typing import Any

from asr_jetson.pipeline.text_backend_contract import (
    TextAnonymizationRequest,
    TextAnonymizationResult,
    canonical_fallback_warning,
)

CANONICAL_BACKEND_CALLABLE = (
    "asr_jetson.postprocessing.transformer_anonymizer.run_transformer_anonymization"
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?:\+?\d[\d .\-]{7,}\d)")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{8,30}\b", re.I)


class TextBackendRuntimeFailure(RuntimeError):
    """Raised when canonical backend execution fails and fallback is not allowed."""


def assert_canonical_callable_path(path: str) -> None:
    if path != CANONICAL_BACKEND_CALLABLE:
        raise ValueError(f"Unsupported backend callable path: {path}")


def normalize_entity_label(label: str) -> str:
    upper = (label or "").upper()
    if upper in {"PER", "PERSON"}:
        return "PERSON"
    if upper in {"ORG", "ORGANIZATION", "COMPANY"}:
        return "ORGANIZATION"
    if upper in {"LOC", "LOCATION", "GPE"}:
        return "LOCATION"
    if "PHONE" in upper or upper in {"TEL", "MOBILE"}:
        return "PHONE"
    if "EMAIL" in upper:
        return "EMAIL"
    if "IBAN" in upper:
        return "IBAN"
    return "MISC"


def _placeholder(label: str, value: str, case_id: str) -> str:
    digest = hmac.new(case_id.encode("utf-8"), value.lower().encode("utf-8"), sha256).hexdigest()[:8]
    return f"<{label}_{digest.upper()}>"


def _sanitize_runtime_failure(prefix: str, exc: Exception) -> str:
    exc_type = type(exc).__name__
    return f"{prefix} ({exc_type}). Check backend dependencies/model availability."


def _regex_only_fallback(
    text: str,
    *,
    case_id: str,
    domain_entities: dict[str, list[str]] | None,
) -> tuple[str, dict[str, Any]]:
    replacements: list[tuple[int, int, str, str]] = []
    for label, pattern in (("EMAIL", _EMAIL_RE), ("PHONE", _PHONE_RE), ("IBAN", _IBAN_RE)):
        for match in pattern.finditer(text):
            replacements.append((match.start(), match.end(), label, match.group(0)))

    for label, values in (domain_entities or {}).items():
        normalized_label = normalize_entity_label(label)
        for value in values:
            if not value:
                continue
            for match in re.finditer(rf"\b{re.escape(value)}\b", text, flags=re.I):
                replacements.append((match.start(), match.end(), normalized_label, match.group(0)))

    replacements.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    deduped: list[tuple[int, int, str, str]] = []
    cursor = -1
    for start, end, label, value in replacements:
        if start < cursor:
            continue
        deduped.append((start, end, label, value))
        cursor = end

    key_to_placeholder: dict[tuple[str, str], str] = {}
    entities: dict[str, dict[str, Any]] = {}
    reverse_map: dict[str, str] = {}
    stats_total = 0
    stats_by_type: dict[str, int] = {}

    anonymized = text
    for start, end, label, value in sorted(deduped, key=lambda item: item[0], reverse=True):
        key = (label, value.lower())
        token = key_to_placeholder.get(key)
        if token is None:
            token = _placeholder(label, value, case_id)
            key_to_placeholder[key] = token
            entities[token] = {
                "label": label,
                "canonical": value,
                "values": [value],
                "pseudonym": token,
            }
            reverse_map[token] = value
            stats_total += 1
            stats_by_type[label] = stats_by_type.get(label, 0) + 1
        anonymized = anonymized[:start] + token + anonymized[end:]

    mapping: dict[str, Any] = {
        "entities": entities,
        "reverse_map": reverse_map,
        "pseudonym_map": {token: token for token in entities},
        "pseudonym_reverse_map": {token: info["canonical"] for token, info in entities.items()},
        "stats": {"total": stats_total, "by_type": stats_by_type},
        "placeholder_style": "regex_only_fallback",
    }
    return anonymized, mapping


def anonymize_text_via_backend(request: TextAnonymizationRequest) -> TextAnonymizationResult:
    case_id = (request.case_id or "default-case").strip() or "default-case"
    if not request.text:
        return TextAnonymizationResult(anonymized_text="", mapping={}, warnings=[], mode="nominal")

    try:
        from asr_jetson.postprocessing import transformer_anonymizer as transformer_backend
    except ImportError:
        anonymized_text, mapping = _regex_only_fallback(
            request.text,
            domain_entities=request.domain_entities,
            case_id=case_id,
        )
        return TextAnonymizationResult(
            anonymized_text=anonymized_text,
            mapping=mapping,
            warnings=[canonical_fallback_warning()],
            mode="degraded_regex_only",
        )

    backend_init_error = getattr(transformer_backend, "TransformerBackendInitializationError", None)
    backend_runtime_error = getattr(transformer_backend, "TransformerBackendRuntimeError", None)

    try:
        anonymized_text, mapping = transformer_backend.run_transformer_anonymization(
            request.text,
            domain_entities=request.domain_entities,
            preserve_dates=request.preserve_dates,
            model_name=request.model_name,
            device=request.device,
        )
        safe_mapping = mapping if isinstance(mapping, dict) else {}
        return TextAnonymizationResult(
            anonymized_text=anonymized_text,
            mapping=safe_mapping,
            warnings=[],
            mode="nominal",
        )
    except Exception as exc:
        if isinstance(exc, ImportError) or (
            isinstance(backend_init_error, type) and isinstance(exc, backend_init_error)
        ):
            anonymized_text, mapping = _regex_only_fallback(
                request.text,
                case_id=case_id,
                domain_entities=request.domain_entities,
            )
            return TextAnonymizationResult(
                anonymized_text=anonymized_text,
                mapping=mapping,
                warnings=[canonical_fallback_warning()],
                mode="degraded_regex_only",
            )

        if isinstance(backend_runtime_error, type) and isinstance(exc, backend_runtime_error):
            raise TextBackendRuntimeFailure(
                _sanitize_runtime_failure(
                    "Text anonymization backend failed during execution",
                    exc,
                )
            ) from exc

        raise TextBackendRuntimeFailure(
            _sanitize_runtime_failure(
                "Unexpected text anonymization backend failure",
                exc,
            )
        ) from exc
