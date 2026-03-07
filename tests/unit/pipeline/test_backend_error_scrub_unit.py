from __future__ import annotations

import pytest

from asr_jetson.pipeline.text_backend_adapter import (
    TextBackendRuntimeFailure,
    anonymize_text_via_backend,
)
from asr_jetson.pipeline.text_backend_contract import TextAnonymizationRequest


@pytest.mark.unit
def test_backend_error_scrub_hides_raw_transcript_fragment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    sensitive_fragment = "TRANSCRIPT_SENSITIVE_FRAGMENT_SHOULD_NOT_LEAK"

    def _raise_runtime(*args, **kwargs):
        raise RuntimeError(f"runtime failure for {sensitive_fragment}")

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _raise_runtime)

    with pytest.raises(TextBackendRuntimeFailure) as exc_info:
        anonymize_text_via_backend(
            TextAnonymizationRequest(
                text=f"Customer said: {sensitive_fragment}",
                case_id="CASE-SCRUB-001",
            )
        )

    message = str(exc_info.value)
    assert sensitive_fragment not in message
    assert "RuntimeError" in message
