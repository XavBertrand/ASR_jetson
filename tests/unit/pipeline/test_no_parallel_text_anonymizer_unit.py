from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_pipeline_does_not_instantiate_transformer_anonymizer_directly() -> None:
    source = Path("src/asr_jetson/pipeline/full_pipeline.py").read_text(encoding="utf-8")
    assert "TransformerAnonymizer(" not in source
    assert "anonymize_with_tags(" not in source
