"""Sanitized logging helpers for anonymization flows."""

from __future__ import annotations

import logging
import re

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?:\+?\d[\d .\-]{7,}\d)")


def sanitize_text(value: str) -> str:
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def sanitize_exception(exc: Exception) -> str:
    return sanitize_text(str(exc))


def get_logger(name: str = "asr_jetson.anonymization") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
