from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.anonymization.core import schema


@pytest.mark.contract
def test_text_backend_feature_keeps_schema_versions_unchanged() -> None:
    assert schema.MAPPING_SCHEMA_VERSION == "1.0"
    assert schema.REPORT_SCHEMA_VERSION == "1.0"


@pytest.mark.contract
def test_text_backend_feature_introduces_no_migration_scripts() -> None:
    migration_candidates = [
        str(path)
        for path in Path("src").rglob("*migration*")
        if path.is_file()
    ]
    assert migration_candidates == []
