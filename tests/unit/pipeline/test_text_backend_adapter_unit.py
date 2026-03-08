from __future__ import annotations

import pytest

from asr_jetson.pipeline.text_backend_adapter import (
    TextBackendRuntimeFailure,
    anonymize_text_via_backend,
)
from asr_jetson.pipeline.text_backend_contract import TextAnonymizationRequest


@pytest.mark.unit
def test_adapter_nominal_uses_canonical_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    def _fake_run(*args, **kwargs):
        return (
            "Alice Bertrand",
            {
                "entities": {
                    "<PERSON_001>": {
                        "label": "PERSON",
                        "canonical": "Alice Martin",
                        "values": ["Alice Martin"],
                    }
                },
                "reverse_map": {"<PERSON_001>": "Alice Martin"},
                "pseudonym_map": {"<PERSON_001>": "<PERSON_001>"},
                "pseudonym_reverse_map": {"<PERSON_001>": "Alice Martin"},
                "stats": {"total": 1, "by_type": {"PERSON": 1}},
            },
        )

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _fake_run)

    result = anonymize_text_via_backend(
        TextAnonymizationRequest(text="Alice Martin", case_id="CASE-001")
    )

    assert result.mode == "nominal"
    assert result.warnings == []
    assert result.anonymized_text == "Alice Bertrand"
    assert result.mapping["pseudonym_map"]["<PERSON_001>"] == "<PERSON_001>"


@pytest.mark.unit
def test_adapter_falls_back_on_backend_initialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    def _raise_init_failure(*args, **kwargs):
        raise transformer_module.TransformerBackendInitializationError("init failed")

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _raise_init_failure)

    result = anonymize_text_via_backend(
        TextAnonymizationRequest(
            text="Contact alice.martin@example.com",
            case_id="CASE-FALLBACK",
        )
    )

    assert result.mode == "degraded_regex_only"
    assert result.warnings
    assert result.warnings[0].warning_code == "NER_UNAVAILABLE_REGEX_FALLBACK"
    assert "alice.martin@example.com" not in result.anonymized_text
    assert "<EMAIL_" in result.anonymized_text


@pytest.mark.unit
def test_adapter_hard_fails_on_runtime_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("runtime exploded for alice.martin@example.com")

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _raise_runtime)

    with pytest.raises(TextBackendRuntimeFailure) as exc_info:
        anonymize_text_via_backend(
            TextAnonymizationRequest(text="secret alice.martin@example.com", case_id="CASE-RUNTIME")
        )

    assert "alice.martin@example.com" not in str(exc_info.value)
