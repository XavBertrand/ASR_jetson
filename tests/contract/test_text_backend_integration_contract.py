from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.pipeline.text_backend_adapter import CANONICAL_BACKEND_CALLABLE


@pytest.mark.contract
def test_contract_declares_canonical_callable() -> None:
    contract = Path("specs/001-transformer-text-backend/contracts/text-backend-integration.md").read_text(
        encoding="utf-8"
    )
    assert CANONICAL_BACKEND_CALLABLE in contract
    assert "MUST NOT invoke any parallel text anonymizer backend" in contract
