from __future__ import annotations

import pytest

from asr_jetson.pipeline.text_backend_adapter import anonymize_text_via_backend
from asr_jetson.pipeline.text_backend_contract import TextAnonymizationRequest


@pytest.mark.unit
def test_import_or_init_failure_uses_regex_fallback_and_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    def _raise_init(*args, **kwargs):
        raise transformer_module.TransformerBackendInitializationError("init failed")

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _raise_init)

    req = TextAnonymizationRequest(
        text="Alice Martin alice.martin@example.com",
        case_id="CASE-FALLBACK-U",
    )

    first = anonymize_text_via_backend(req)
    second = anonymize_text_via_backend(req)

    assert first.mode == "degraded_regex_only"
    assert second.mode == "degraded_regex_only"
    assert first.anonymized_text == second.anonymized_text
    assert first.warnings and first.warnings[0].warning_code == "NER_UNAVAILABLE_REGEX_FALLBACK"
