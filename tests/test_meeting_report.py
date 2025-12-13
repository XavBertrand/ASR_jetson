from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asr_jetson.postprocessing import meeting_report, mistral_client  # noqa: E402

try:  # pragma: no cover - optional dependency
    import pypandoc  # type: ignore
except Exception:  # pragma: no cover - executed when pypandoc missing
    pypandoc = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from weasyprint import HTML  # type: ignore
except Exception:  # pragma: no cover - executed when weasyprint missing
    HTML = None  # type: ignore


def _missing_prereqs() -> list[str]:
    missing: list[str] = []
    if pypandoc is None:
        missing.append("pypandoc")
    else:  # pragma: no cover - trivial branch
        try:
            pypandoc.get_pandoc_version()  # type: ignore[attr-defined]
        except OSError as exc:  # pragma: no cover - executed when pandoc binary missing
            missing.append(f"pandoc ({exc})")
    if HTML is None:
        missing.append("weasyprint")
    return missing


_MISSING = _missing_prereqs()
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason="Missing prerequisites: " + ", ".join(_MISSING),
)


def _pseudo_replace(text: str, mapping: dict) -> str:
    """
    Roughly anonymize the text by swapping canonicals to pseudonyms based on the mapping.
    """
    reverse = mapping.get("pseudonym_reverse_map") or {}
    canon_to_pseudo = {canonical: pseudo for pseudo, canonical in reverse.items() if canonical}
    anonymized = text
    for canonical, pseudo in sorted(
        canon_to_pseudo.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        anonymized = re.sub(rf"\b{re.escape(canonical)}\b", pseudo, anonymized)
    return anonymized


def test_pipeline_style_meeting_report_generation(tmp_path: Path, monkeypatch):
    transcription_path = PROJECT_ROOT / "tests/data/transcription.txt"
    mapping_path = PROJECT_ROOT / "tests/data/meeting_anon_mapping.json"
    anon_report_fixture = (
        PROJECT_ROOT / "tests/data/integration_real_call_meeting_report_anonymized.txt"
    )
    if (
        not transcription_path.exists()
        or not mapping_path.exists()
        or not anon_report_fixture.exists()
    ):
        pytest.skip("Fixtures manquantes pour le test de rapport de réunion")

    raw_text = transcription_path.read_text(encoding="utf-8")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    # 1) Anonymisation simplifiée pour produire une transcription anonymisée.
    anonymized_text = _pseudo_replace(raw_text, mapping)

    # 2) Appel Mistral (mocké) pour générer le rapport anonymisé.
    anonymized_report = anon_report_fixture.read_text(encoding="utf-8")
    monkeypatch.setattr(
        mistral_client,
        "chat_complete",
        lambda model, system, user_text, temperature=None: anonymized_report,
    )
    prompts_path = PROJECT_ROOT / "src/asr_jetson/config/mistral_prompts.json"
    prompt = mistral_client.load_prompts(str(prompts_path))
    meeting_date = "2024-01-02"
    generated_anonymized_report = mistral_client.chat_complete(
        prompt.model,
        prompt.system,
        prompt.user_prefix.format(meeting_date=meeting_date) + anonymized_text,
        temperature=0.1,
    )

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    anonymized_report_path = reports_dir / "transcription_meeting_report_anonymized.md"
    anonymized_report_path.write_text(generated_anonymized_report, encoding="utf-8")

    # 3) Génération du PDF via meeting_report (désanonymisation + export HTML/PDF).
    result = meeting_report.generate_pdf_report(
        anonymized_markdown_path=anonymized_report_path,
        mapping_json_path=mapping_path,
        output_dir=tmp_path,
        run_id="transcription",
        title="Entretien professionnel",
        meeting_date=meeting_date,
        audio_stem="transcription",
        run_time="101112",
    )

    md_path = Path(result["report_markdown"])
    pdf_path = Path(result["report_pdf"])
    docx_path = Path(result["report_docx"])
    assert md_path.exists(), "Le Markdown désanonymisé doit être écrit"
    assert pdf_path.exists(), "Le PDF doit être généré"
    assert docx_path.exists(), "Le DOCX doit être généré"
    assert docx_path.suffix == ".docx"

    markdown = md_path.read_text(encoding="utf-8")
    # Les pseudonymes doivent avoir été remplacés.
    assert "Bleu Horizon Partners" not in markdown
    assert "Bleu Horizon Industries" not in markdown
    assert "Pauline Blanc" not in markdown
    assert "Udaf" in markdown
    assert "Detail Group" in markdown
    assert "Marine" in markdown
    assert "| Pseudonyme" in markdown
    assert markdown.count("###") >= 6

    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF"), "Le fichier généré doit être un PDF valide"
    assert pdf_path.stat().st_size > 500, "Le PDF doit contenir du contenu"
    assert docx_path.stat().st_size > 500, "Le DOCX doit contenir du contenu"


def test_end_to_end_report_generation(tmp_path: Path, monkeypatch):
    """
    Chemin complet : transcription brute -> anonymisation simple -> Mistral (mock)
    puis PDF désanonymisé.
    """
    transcription_path = PROJECT_ROOT / "tests/data/transcription.txt"
    mapping_path = PROJECT_ROOT / "tests/data/meeting_anon_mapping.json"
    report_fixture = PROJECT_ROOT / "tests/data/integration_real_call_meeting_report_anonymized.txt"
    if not transcription_path.exists() or not mapping_path.exists() or not report_fixture.exists():
        pytest.skip("Fixtures manquantes pour le test de rapport de réunion")

    raw_text = transcription_path.read_text(encoding="utf-8")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    # 1) Anonymisation (remplacement des canonicals par leurs pseudonymes).
    anonymized_transcription = _pseudo_replace(raw_text, mapping)
    anon_txt_path = tmp_path / "txt" / "transcription_anon.txt"
    anon_txt_path.parent.mkdir(parents=True, exist_ok=True)
    anon_txt_path.write_text(anonymized_transcription, encoding="utf-8")

    # 2) Mistral (mock) pour produire le rapport anonymisé.
    anonymized_report_md = report_fixture.read_text(encoding="utf-8")
    captured: dict[str, str] = {}
    meeting_date = "2024-01-03"

    def _fake_chat_complete(model, system, user_text, temperature=None):
        captured["model"] = model
        captured["system"] = system
        captured["user_text"] = user_text
        return anonymized_report_md

    monkeypatch.setattr(mistral_client, "chat_complete", _fake_chat_complete)
    prompt = mistral_client.load_prompts(
        str(PROJECT_ROOT / "src/asr_jetson/config/mistral_prompts.json")
    )
    generated_report = mistral_client.chat_complete(
        prompt.model,
        prompt.system,
        prompt.user_prefix.format(meeting_date=meeting_date) + anonymized_transcription,
        temperature=0.1,
    )
    assert captured.get("user_text", "").startswith(
        prompt.user_prefix.format(meeting_date=meeting_date)
    ), "Le prompt utilisateur doit être préfixé"
    assert captured.get("model") == prompt.model
    assert captured.get("system") == prompt.system
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    anonymized_report_path = reports_dir / "transcription_meeting_report_anonymized.md"
    anonymized_report_path.write_text(generated_report, encoding="utf-8")

    # 3) Désanonymisation + export PDF.
    result = meeting_report.generate_pdf_report(
        anonymized_markdown_path=anonymized_report_path,
        mapping_json_path=mapping_path,
        output_dir=tmp_path,
        run_id="transcription",
        title="Entretien professionnel",
        meeting_date=meeting_date,
        audio_stem="transcription",
        run_time="131415",
    )

    md_path = Path(result["report_markdown"])
    pdf_path = Path(result["report_pdf"])
    docx_path = Path(result["report_docx"])
    assert md_path.exists(), "Le Markdown désanonymisé doit être produit"
    assert pdf_path.exists(), "Le PDF doit être généré"
    assert docx_path.exists(), "Le DOCX doit être généré"

    markdown = md_path.read_text(encoding="utf-8")
    assert "Bleu Horizon Partners" not in markdown
    assert "Montfleury Conseil" not in markdown
    assert "Lucas Adam" not in markdown
    assert "Migmeca" in markdown
    assert "Detail Group" in markdown
    assert "Udaf" in markdown
    assert "Marine" in markdown
    assert "| Pseudonyme" in markdown
    assert markdown.count("###") >= 6

    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_path.stat().st_size > 500
    assert docx_path.stat().st_size > 500
