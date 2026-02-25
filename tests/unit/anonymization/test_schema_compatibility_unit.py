from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.anonymization.core.errors import InputValidationError
from asr_jetson.anonymization.storage.compat import load_versioned_json


@pytest.mark.unit
def test_load_versioned_json_accepts_supported_schema(tmp_path: Path) -> None:
    payload = {"schema_version": "1.0", "value": 123}
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_versioned_json(path, {"1.0", "1.1"})
    assert loaded["value"] == 123


@pytest.mark.unit
def test_load_versioned_json_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"schema_version": "2.0"}), encoding="utf-8")

    with pytest.raises(InputValidationError) as exc:
        load_versioned_json(path, {"1.0", "1.1"})

    assert exc.value.code == "INPUT_VALIDATION_ERROR"
    assert "Unsupported schema version" in exc.value.message_safe
