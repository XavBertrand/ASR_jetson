from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_jetson.postprocessing import meeting_report, mistral_client


@pytest.mark.unit
def test_load_prompts_fallback_key(tmp_path) -> None:
    prompts = {
        "entretien_collaborateur": {
            "model": "m1",
            "system": "sys",
            "user_prefix": "prefix",
        }
    }
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(prompts), encoding="utf-8")

    prompt = mistral_client.load_prompts(str(path), key="meeting_analysis")
    assert prompt.model == "m1"
    assert prompt.system == "sys"


@pytest.mark.unit
def test_load_prompts_unknown_key_raises(tmp_path) -> None:
    prompts = {
        "entretien_collaborateur": {
            "model": "m1",
            "system": "sys",
            "user_prefix": "prefix",
        }
    }
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(prompts), encoding="utf-8")

    with pytest.raises(KeyError):
        mistral_client.load_prompts(str(path), key="missing")


@pytest.mark.unit
def test_normalize_mapping_dict_entities() -> None:
    mapping = {
        "entities": {
            "<PER_1>": {
                "label": "PERSON",
                "canonical": "Alice",
                "values": ["Alice", "A."],
            }
        }
    }
    normalized = meeting_report._normalize_mapping(mapping)
    assert isinstance(normalized["entities"], list)
    assert normalized["entities"][0]["tag"] == "<PER_1>"


@pytest.mark.unit
def test_derive_base_name_and_safe_component() -> None:
    assert (
        meeting_report._derive_base_name(Path("/tmp/report_anon_clean.md"), run_id=None)
        == "report"
    )
    assert (
        meeting_report._derive_base_name(Path("/tmp/report_anonymized.md"), run_id="run-1")
        == "run-1"
    )

    assert meeting_report._safe_filename_component("bad/dir", "fallback") == "bad_dir"
    assert meeting_report._safe_filename_component("", "fallback") == "fallback"


@pytest.mark.unit
def test_resolve_default_report_title() -> None:
    assert meeting_report.resolve_default_report_title("entretien_collaborateur")
    assert (
        meeting_report.resolve_default_report_title("missing")
        == meeting_report.DEFAULT_REPORT_TITLE
    )
    assert meeting_report.resolve_default_report_title(None) == meeting_report.DEFAULT_REPORT_TITLE


@pytest.mark.unit
def test_polish_markdown_with_languagetool(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMatch:
        def __init__(self, offset: int, error_length: int, replacements: list[str]):
            self.offset = offset
            self.errorLength = error_length
            self.replacements = replacements

    class DummyTool:
        def check(self, _text: str):
            return [
                DummyMatch(offset=7, error_length=2, replacements=[" "]),
                DummyMatch(offset=9, error_length=5, replacements=["Alicia"]),
            ]

    monkeypatch.setattr(meeting_report, "_ensure_language_tool", lambda: DummyTool())

    text = "Bonjour  Alice"
    cleaned = meeting_report._polish_markdown_with_languagetool(text)

    assert cleaned == "Bonjour Alice"
