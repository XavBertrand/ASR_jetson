from __future__ import annotations

import pytest

from asr_jetson.pipeline.text_backend_adapter import anonymize_text_via_backend
from asr_jetson.pipeline.text_backend_contract import TextAnonymizationRequest


def _request(case_id: str) -> TextAnonymizationRequest:
    return TextAnonymizationRequest(
        text="Alice Martin - alice.martin@example.com",
        case_id=case_id,
    )


@pytest.mark.integration
def test_nominal_mode_same_case_deterministic_and_cross_case_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    def _fake_nominal(*args, **kwargs):
        return (
            "<PERSON_001> <EMAIL_001>",
            {
                "entities": {
                    "<PERSON_001>": {"label": "PERSON", "canonical": "Alice Martin", "values": ["Alice Martin"]},
                    "<EMAIL_001>": {"label": "EMAIL", "canonical": "alice.martin@example.com", "values": ["alice.martin@example.com"]},
                },
                "reverse_map": {
                    "<PERSON_001>": "Alice Martin",
                    "<EMAIL_001>": "alice.martin@example.com",
                },
                "pseudonym_map": {
                    "<PERSON_001>": "<PERSON_001>",
                    "<EMAIL_001>": "<EMAIL_001>",
                },
                "pseudonym_reverse_map": {
                    "<PERSON_001>": "Alice Martin",
                    "<EMAIL_001>": "alice.martin@example.com",
                },
                "stats": {"total": 2, "by_type": {"PERSON": 1, "EMAIL": 1}},
            },
        )

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _fake_nominal)

    same_a = anonymize_text_via_backend(_request("CASE-A"))
    same_b = anonymize_text_via_backend(_request("CASE-A"))
    other = anonymize_text_via_backend(_request("CASE-B"))

    assert same_a.anonymized_text == same_b.anonymized_text
    assert same_a.mapping == same_b.mapping
    assert same_a.anonymized_text != other.anonymized_text


@pytest.mark.integration
def test_fallback_mode_same_case_deterministic_and_cross_case_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asr_jetson.postprocessing import transformer_anonymizer as transformer_module

    def _raise_init(*args, **kwargs):
        raise transformer_module.TransformerBackendInitializationError("init failed")

    monkeypatch.setattr(transformer_module, "run_transformer_anonymization", _raise_init)

    same_a = anonymize_text_via_backend(_request("CASE-A"))
    same_b = anonymize_text_via_backend(_request("CASE-A"))
    other = anonymize_text_via_backend(_request("CASE-B"))

    assert same_a.anonymized_text == same_b.anonymized_text
    assert same_a.mapping == same_b.mapping
    assert same_a.anonymized_text != other.anonymized_text
    assert same_a.warnings and same_a.warnings[0].warning_code == "NER_UNAVAILABLE_REGEX_FALLBACK"
