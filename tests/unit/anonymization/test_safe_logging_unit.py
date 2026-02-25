from __future__ import annotations

import logging

import pytest

from asr_jetson.anonymization.core.safe_logging import (
    get_logger,
    log_safe,
    sanitize_mapping,
    sanitize_text,
)


@pytest.mark.unit
def test_sanitize_text_redacts_sensitive_patterns() -> None:
    raw = "Email alice.martin@example.com phone +33 06 11 22 33 44 IBAN FR761234567890"
    cleaned = sanitize_text(raw)

    assert "alice.martin@example.com" not in cleaned
    assert "+33 06 11 22 33 44" not in cleaned
    assert "FR761234567890" not in cleaned
    assert "[REDACTED_EMAIL]" in cleaned


@pytest.mark.unit
def test_log_safe_redacts_fields(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("asr_jetson.anonymization.test.safe")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_safe(
            logger,
            logging.INFO,
            "processing alice.martin@example.com",
            contact="alice.martin@example.com",
            phone="+33 06 11 22 33 44",
            nested={"email": "alice.martin@example.com"},
        )

    rendered = caplog.text
    assert "alice.martin@example.com" not in rendered
    assert "+33 06 11 22 33 44" not in rendered
    assert "[REDACTED_EMAIL]" in rendered


@pytest.mark.unit
def test_sanitize_mapping_handles_nested_values() -> None:
    payload = {
        "email": "alice.martin@example.com",
        "nested": {"phone": "+33 06 11 22 33 44"},
        "list": ["alice.martin@example.com", {"iban": "FR761234567890"}],
    }

    cleaned = sanitize_mapping(payload)
    assert "alice.martin@example.com" not in str(cleaned)
    assert "+33 06 11 22 33 44" not in str(cleaned)
    assert "FR761234567890" not in str(cleaned)
