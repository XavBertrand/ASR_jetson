"""Sanitized logging helpers for anonymization flows."""

from __future__ import annotations

import logging
import re

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?:\+?\d[\d .\-]{7,}\d)")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{8,30}\b", re.I)
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def sanitize_text(value: str) -> str:
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = _IBAN_RE.sub("[REDACTED_IBAN]", text)
    text = _CARD_RE.sub("[REDACTED_CARD]", text)
    return text


def sanitize_exception(exc: Exception) -> str:
    return sanitize_text(str(exc))


def sanitize_mapping(payload: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_text(value)
        elif isinstance(value, dict):
            nested = {str(k): v for k, v in value.items()}
            sanitized[key] = sanitize_mapping(nested)
        elif isinstance(value, list):
            cleaned: list[object] = []
            for item in value:
                if isinstance(item, str):
                    cleaned.append(sanitize_text(item))
                elif isinstance(item, dict):
                    cleaned.append(sanitize_mapping({str(k): v for k, v in item.items()}))
                else:
                    cleaned.append(item)
            sanitized[key] = cleaned
        else:
            sanitized[key] = value
    return sanitized


def log_safe(logger: logging.Logger, level: int, message: str, **fields: object) -> None:
    safe_fields = sanitize_mapping(fields)
    rendered_fields = " ".join(f"{k}={v}" for k, v in safe_fields.items())
    safe_message = sanitize_text(message)
    logger.log(level, f"{safe_message} {rendered_fields}".strip())


def get_logger(name: str = "asr_jetson.anonymization") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
