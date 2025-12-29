from __future__ import annotations

import pytest

from asr_jetson.postprocessing import anonymizer


@pytest.mark.unit
def test_normalize_ollama_base_url() -> None:
    assert (
        anonymizer.normalize_ollama_base_url("http://localhost:11434/v1")
        == "http://localhost:11434"
    )
    assert anonymizer.normalize_ollama_base_url("") == "http://localhost:11434"


@pytest.mark.unit
def test_normalize_text_titles_and_hyphens() -> None:
    raw = "M.Dupont s\u2019appelle Jean\u2013Paul"
    cleaned = anonymizer.normalize_text(raw)
    assert "M. Dupont" in cleaned
    assert "Jean-Paul" in cleaned


@pytest.mark.unit
def test_split_sentences() -> None:
    text = "Bonjour. Salut !\nMerci."
    parts = anonymizer.split_sentences(text)
    assert parts == ["Bonjour.", "Salut !", "Merci."]


@pytest.mark.unit
def test_make_blocks_groups_sentences() -> None:
    text = "Bonjour. Salut ! Merci."
    sentences = anonymizer.split_sentences(text)
    blocks = anonymizer.make_blocks(text, sentences, max_chars=50, max_sents=2)
    assert len(blocks) == 2
    assert blocks[0]["text"].startswith("Bonjour.")


@pytest.mark.unit
def test_spans_from_regex() -> None:
    text = "Contact: test@example.com ou +33 6 12 34 56 78."
    spans = anonymizer.spans_from_regex(text)
    types = {span["type"] for span in spans}
    assert "EMAIL" in types
    assert "PHONE" in types


@pytest.mark.unit
def test_spans_from_ner_thresholds() -> None:
    cfg = anonymizer.Settings(
        per_threshold=0.7,
        org_threshold=0.7,
        loc_threshold=0.7,
        other_threshold=0.8,
    )
    ner_outputs = [
        {"entity_group": "PER", "score": 0.8, "start": 0, "end": 4},
        {"entity_group": "ORG", "score": 0.6, "start": 5, "end": 9},
        {"entity_group": "MISC", "score": 0.9, "start": 10, "end": 12},
    ]
    spans = anonymizer.spans_from_ner(ner_outputs, cfg)
    types = {span["type"] for span in spans}
    assert "PER" in types
    assert "MISC" in types
    assert "ORG" not in types
