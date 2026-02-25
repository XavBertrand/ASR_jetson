"""Schema compatibility helpers for report and mapping artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from asr_jetson.anonymization.core.errors import InputValidationError


def load_versioned_json(path: Path, supported_versions: set[str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = str(payload.get("schema_version", ""))
    if version not in supported_versions:
        raise InputValidationError(f"Unsupported schema version: {version}")
    return payload
