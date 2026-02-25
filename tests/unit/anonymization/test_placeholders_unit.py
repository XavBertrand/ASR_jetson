from __future__ import annotations

import pytest

from asr_jetson.anonymization.core.placeholders import PlaceholderGenerator


@pytest.mark.unit
def test_placeholders_are_deterministic_within_case() -> None:
    generator = PlaceholderGenerator(case_id="CASE-DET-001", schema_version="1.0", secret=b"secret")

    token_1 = generator.generate("person", "Alice Martin")
    token_2 = generator.generate("person", "Alice Martin")
    token_3 = generator.generate("person", "  Alice   Martin  ")

    assert token_1 == token_2
    assert token_1 == token_3


@pytest.mark.unit
def test_placeholders_are_isolated_across_cases() -> None:
    generator_a = PlaceholderGenerator(case_id="CASE-A", schema_version="1.0", secret=b"secret")
    generator_b = PlaceholderGenerator(case_id="CASE-B", schema_version="1.0", secret=b"secret")

    token_a = generator_a.generate("person", "Alice Martin")
    token_b = generator_b.generate("person", "Alice Martin")

    assert token_a != token_b
