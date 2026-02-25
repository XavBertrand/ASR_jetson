"""Deterministic placeholder generation scoped by case_id."""

from __future__ import annotations

import hashlib
import hmac
import os
import re


_NORMALIZE_WS = re.compile(r"\s+")


def normalize_entity_value(value: str) -> str:
    cleaned = _NORMALIZE_WS.sub(" ", value.strip().lower())
    return cleaned


class PlaceholderGenerator:
    """Generate deterministic placeholders isolated by case_id."""

    def __init__(self, case_id: str, schema_version: str = "1.0", secret: bytes | None = None) -> None:
        if not case_id.strip():
            raise ValueError("case_id must not be empty")
        self.case_id = case_id
        self.schema_version = schema_version
        configured_secret = secret
        if configured_secret is None:
            raw = os.environ.get("ANON_PLACEHOLDER_SECRET", "asr-jetson-placeholder-secret")
            configured_secret = raw.encode("utf-8")
        # Derive per-case key to ensure cross-case namespace isolation.
        self._case_key = hmac.new(configured_secret, case_id.encode("utf-8"), hashlib.sha256).digest()

    def generate(self, entity_type: str, entity_value: str) -> str:
        normalized = normalize_entity_value(entity_value)
        payload = f"{self.schema_version}|{entity_type.lower()}|{normalized}".encode("utf-8")
        digest = hmac.new(self._case_key, payload, hashlib.sha256).hexdigest().upper()
        prefix = entity_type.upper().replace(" ", "_")
        return f"<{prefix}_{digest[:12]}>"
