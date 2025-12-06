from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

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


def _integration_prerequisites() -> list[str]:
    missing: list[str] = []
    if not os.getenv("MISTRAL_API_KEY"):
        missing.append("MISTRAL_API_KEY")
    if not _HAS_PYPANDOC:
        missing.append("pypandoc")
    return missing


_INTEGRATION_MISSING = _integration_prerequisites()
_SKIP_INTEGRATION = bool(_INTEGRATION_MISSING)
_SKIP_REASON = (
    "Missing integration prerequisites: " + ", ".join(_INTEGRATION_MISSING)
    if _INTEGRATION_MISSING
    else ""
)


def _get_transcription_path() -> Path:
    path = PROJECT_ROOT / "tests/data/transcription.txt"
    if not path.exists():
        pytest.skip(f"Fixture manquante: {path}")
    return path


def _collect_canonical_entities(mapping: dict) -> set[str]:
    names: set[str] = set()
    entities = mapping.get("entities", [])
    if isinstance(entities, dict):
        reverse = mapping.get("reverse_map", {}) or {}
        for tag, value in reverse.items():
            if value and "PERSON" in str(tag).upper():
                names.add(value)
        for tag, info in entities.items():
            if "PERSON" not in str(info.get("label", "")).upper() and "PERSON" not in str(tag).upper():
                continue
            for value in info.get("values", []):
                if value:
                    names.add(value)
    else:
        for entity in entities:
            if str(entity.get("type") or entity.get("label") or "").upper() != "PERSON":
                continue
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


def test_transcription_to_markdown_offline(tmp_path: Path, monkeypatch):
    """
    Génère un rapport Markdown à partir de la transcription fixture sans dépendance
    réseau (LLM mocké) ni pypandoc.
    """
    transcription_path = _get_transcription_path()
    raw_text = transcription_path.read_text(encoding="utf-8")

    outputs_root = tmp_path / "outputs"
    txt_dir = outputs_root / "txt"
    json_dir = outputs_root / "json"
    txt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = PROJECT_ROOT / "src/asr_jetson/config/mistral_prompts.json"

    # On part de la transcription brute et on injecte des tags simplifiés pour le test.
    anonymized_text = raw_text + "\n\nParticipants: <PERSON_1>, <PERSON_2>\n"
    anon_path = txt_dir / "transcription_offline_anon_clean.txt"
    anon_path.write_text(anonymized_text, encoding="utf-8")

    mapping = {
        "entities": {
            "<PERSON_1>": {
                "label": "PERSON",
                "values": ["Delphine"],
                "canonical": "Delphine",
                "pseudonym": "Alice Dupont",
                "source": "stub",
            },
            "<PERSON_2>": {
                "label": "PERSON",
                "values": ["Marine"],
                "canonical": "Marine",
                "pseudonym": "Brigitte Durand",
                "source": "stub",
            },
        },
        "reverse_map": {"<PERSON_1>": "Delphine", "<PERSON_2>": "Marine"},
        "pseudonym_map": {"<PERSON_1>": "Alice Dupont", "<PERSON_2>": "Brigitte Durand"},
        "pseudonym_reverse_map": {"Alice Dupont": "Delphine", "Brigitte Durand": "Marine"},
    }
    mapping_path = json_dir / "transcription_offline_anon_mapping.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    fake_report = (
        "### RÉSUMÉ EXÉCUTIF\n"
        "Synthèse courte des échanges.\n\n"
        "### PARTICIPANTS\n"
        "| Pseudonyme | Désignation | Catégorie | Indices contextuels | Genre |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| <PERSON_1> | Collaborateur | Collaborateur | - Gère les dossiers clients | Féminin |\n"
        "| <PERSON_2> | Avocat gérante | Avocat gérante | - Prend les décisions stratégiques | Féminin |\n"
    )

    monkeypatch.setattr(meeting_report_mod, "_check_mistral_access", lambda timeout=5.0: (True, ""))
    monkeypatch.setattr(
        meeting_report_mod.mistral_client,
        "chat_complete",
        lambda model, system, user_text: fake_report,
    )

    def _fake_convert(markdown_text: str, to: str, out_path: Path) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if to == "pdf":
            out.write_bytes(b"%PDF-1.4\\n1 0 obj<<>>\\nendobj\\ntrailer<< /Size 1 >>\\n%%EOF\\n")
        else:
            out.write_text(f"{to} export stub\\n{markdown_text}", encoding="utf-8")

    monkeypatch.setattr(meeting_report_mod, "_convert_markdown_with_pandoc", _fake_convert)

    result = meeting_report_mod.generate_meeting_report(
        anonymized_txt_path=anon_path,
        mapping_json_path=mapping_path,
        prompts_json_path=prompts_path,
        prompt_key="meeting_analysis",
        out_dir=txt_dir,
        run_id="transcription_offline",
    )

    report_md_path = Path(result["report_markdown"])
    assert report_md_path.exists(), "Le rapport Markdown doit être généré"
    markdown = report_md_path.read_text(encoding="utf-8")
    assert "| Pseudonyme" in markdown
    assert "| --- | --- |" in markdown
    assert "\n|\n" not in markdown  # pas de lignes '|' orphelines


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_transcription_to_docx_meeting_report(tmp_path: Path):
    transcription_path = _get_transcription_path()
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
    present = {name for name in expected_names if name.lower() in dean_report_text.lower()}
    assert present, "Au moins un nom canonique doit apparaître dans le rapport désanonymisé"
    assert len(present) >= min(3, len(expected_names)), (
        f"Noms présents insuffisants ({len(present)}/{len(expected_names)}) : {sorted(expected_names)}"
    )
    assert dean_report_text.count("\n") > 5, "Le rapport doit comporter plusieurs lignes structurées"

    docx_text = _read_docx_text(report_docx_path)
    docx_present = {name for name in expected_names if name.lower() in docx_text.lower()}
    assert docx_present, "Au moins un nom canonique doit apparaître dans le DOCX"
    assert len(docx_present) >= min(3, len(expected_names)), (
        f"Noms DOCX insuffisants ({len(docx_present)}/{len(expected_names)}) : {sorted(expected_names)}"
    )

    assert report_md_path.read_text(encoding="utf-8").startswith("###"), "Le markdown doit conserver la structure"


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_generate_meeting_report_calls_real_mistral(tmp_path: Path):
    outputs_root = tmp_path / "outputs"
    txt_dir = outputs_root / "txt"
    json_dir = outputs_root / "json"
    txt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    raw_path = _get_transcription_path()
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
