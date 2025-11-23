from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import requests  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asr_jetson.postprocessing.transformer_anonymizer import TransformerAnonymizer
import src.asr_jetson.postprocessing.meeting_report as meeting_report_mod

try:
    from docx import Document  # type: ignore
except Exception:  # python-docx absent
    Document = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pypandoc  # type: ignore  # noqa: F401

    _HAS_PYPANDOC = True
except Exception:  # pragma: no cover - executed when pypandoc missing
    _HAS_PYPANDOC = False

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _integration_prerequisites() -> list[str]:
    missing: list[str] = []
    if not os.getenv("MISTRAL_API_KEY"):
        missing.append("MISTRAL_API_KEY")
    if not _HAS_PYPANDOC:
        missing.append("pypandoc")
    try:
        resp = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/version", timeout=5)
        if resp.status_code >= 400:
            missing.append(f"Ollama ({OLLAMA_URL}) status={resp.status_code}")
    except Exception as exc:  # pragma: no cover - depends on environment
        missing.append(f"Ollama ({OLLAMA_URL}) unreachable: {exc}")
    return missing


_INTEGRATION_MISSING = _integration_prerequisites()
_SKIP_INTEGRATION = bool(_INTEGRATION_MISSING)
_SKIP_REASON = (
    "Missing integration prerequisites: " + ", ".join(_INTEGRATION_MISSING)
    if _INTEGRATION_MISSING
    else ""
)


def _collect_canonical_entities(mapping: dict) -> set[str]:
    names: set[str] = set()
    entities = mapping.get("entities", [])
    if isinstance(entities, dict):
        for value in mapping.get("reverse_map", {}).values():
            if value:
                names.add(value)
        for info in entities.values():
            for value in info.get("values", []):
                if value:
                    names.add(value)
    else:
        for entity in entities:
            canonical = entity.get("canonical")
            if canonical:
                names.add(canonical)
    return names


def _read_docx_text(path: Path) -> str:
    if Document is not None:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    with ZipFile(str(path), "r") as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
    return xml


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_transcription_to_docx_meeting_report(tmp_path: Path):
    transcription_path = PROJECT_ROOT / "tests/data/transcription.txt"
    raw_text = transcription_path.read_text(encoding="utf-8")

    outputs_root = tmp_path / "outputs"
    txt_dir = outputs_root / "txt"
    json_dir = outputs_root / "json"
    txt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = PROJECT_ROOT / "src/asr_jetson/config/mistral_prompts.json"

    anonymizer = TransformerAnonymizer()
    anonymized_text, mapping = anonymizer.anonymize_with_tags(raw_text)

    anon_clean_path = txt_dir / "transcription_anon_clean.txt"
    anon_clean_path.write_text(anonymized_text, encoding="utf-8")

    mapping_path = json_dir / "transcription_anon_mapping.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    result = meeting_report_mod.generate_meeting_report(
        anonymized_txt_path=anon_clean_path,
        mapping_json_path=mapping_path,
        prompts_json_path=prompts_path,
        prompt_key="meeting_analysis",
        out_dir=txt_dir,
        run_id="transcription",
    )

    report_docx_path = Path(result["report_docx"])
    report_txt_path = Path(result["report_txt"])
    report_txt_anon_path = Path(result["report_anonymized_txt"])
    report_pdf_path = Path(result["report_pdf"])
    report_md_path = Path(result["report_markdown"])

    for path in [
        report_docx_path,
        report_txt_path,
        report_txt_anon_path,
        report_pdf_path,
        report_md_path,
    ]:
        assert path.exists(), f"{path} devrait être généré"
        assert path.stat().st_size > 0, f"{path} ne doit pas être vide"

    anon_report_text = report_txt_anon_path.read_text(encoding="utf-8")
    pseudonyms = list(mapping.get("pseudonym_reverse_map", {}).keys())
    if pseudonyms:
        assert any(p in anon_report_text for p in pseudonyms), "Le rapport anonymisé doit réutiliser les pseudonymes"

    dean_report_text = report_txt_path.read_text(encoding="utf-8")
    expected_names = _collect_canonical_entities(mapping)
    for name in expected_names:
        assert name in dean_report_text, f"{name} doit être restauré dans le rapport désanonymisé"
    assert dean_report_text.count("\n") > 5, "Le rapport doit comporter plusieurs lignes structurées"

    docx_text = _read_docx_text(report_docx_path)
    for name in expected_names:
        assert name in docx_text

    assert report_md_path.read_text(encoding="utf-8").startswith("###"), "Le markdown doit conserver la structure"


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_generate_meeting_report_calls_real_mistral(tmp_path: Path):
    outputs_root = tmp_path / "outputs"
    txt_dir = outputs_root / "txt"
    json_dir = outputs_root / "json"
    txt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    raw_path = PROJECT_ROOT / "tests/data/transcription.txt"
    raw_text = raw_path.read_text(encoding="utf-8")

    anonymizer = TransformerAnonymizer()
    anon_text, mapping = anonymizer.anonymize_with_tags(raw_text)

    anonymized_txt_path = txt_dir / "meeting_anon_clean.txt"
    mapping_json_path = json_dir / "meeting_anon_mapping.json"
    anonymized_txt_path.write_text(anon_text, encoding="utf-8")
    mapping_json_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    prompts_path = PROJECT_ROOT / "src/asr_jetson/config/mistral_prompts.json"

    try:
        result = meeting_report_mod.generate_meeting_report(
            anonymized_txt_path=anonymized_txt_path,
            mapping_json_path=mapping_json_path,
            prompts_json_path=prompts_path,
            prompt_key="meeting_analysis",
            out_dir=txt_dir,
            run_id="integration_real_call",
        )
    except Exception as exc:  # pragma: no cover - depends on external service state
        message = str(exc)
        if "Service tier capacity exceeded" in message or "429" in message:
            pytest.skip(f"Mistral API throttled the request: {message}")
        raise

    report_docx_path = Path(result["report_docx"])
    report_txt_path = Path(result["report_txt"])
    report_txt_anon_path = Path(result["report_anonymized_txt"])
    report_pdf_path = Path(result["report_pdf"])
    report_md_path = Path(result["report_markdown"])

    for path in [
        report_docx_path,
        report_txt_path,
        report_txt_anon_path,
        report_pdf_path,
        report_md_path,
    ]:
        assert path.exists(), f"{path} devrait être généré"

    deanon_text = report_txt_path.read_text(encoding="utf-8")
    assert len(deanon_text.strip()) > 50, "Le rapport doit contenir une analyse substantielle"
