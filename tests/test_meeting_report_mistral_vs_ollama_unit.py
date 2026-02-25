from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests

from asr_jetson.postprocessing import mistral_client
from asr_jetson.postprocessing.meeting_report import (
    format_meeting_date_literal,
    generate_pdf_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_TRANSCRIPTION_PATH = (
    PROJECT_ROOT
    / "../ASR_action_avocats/recordings/Delphine/runs/"
    / "entretien_el_moussaoui_20251229T152947Z_20260203T101956Z_20260203T101956Z/"
    / "txt/entretien_el_moussaoui_20251229T152947Z_20260203T101956Z_pyannote_"
    "speaker-diarization-3.1_large-v3_clean.txt"
)
REAL_META_PATH = (
    PROJECT_ROOT
    / "../ASR_action_avocats/recordings/Delphine/runs/"
    / "entretien_el_moussaoui_20251229T152947Z_20260203T101956Z_20260203T101956Z/meta.json"
)


def _skip_if_missing_real_inputs() -> None:
    if not REAL_TRANSCRIPTION_PATH.exists():
        pytest.skip(f"Transcription réelle introuvable: {REAL_TRANSCRIPTION_PATH}")
    if not REAL_META_PATH.exists():
        pytest.skip(f"Meta réel introuvable: {REAL_META_PATH}")


def _skip_if_missing_report_prereqs() -> None:
    try:
        import pypandoc  # type: ignore

        pypandoc.get_pandoc_version()  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - depends on local environment
        pytest.skip(f"Prérequis rapport manquant: pypandoc/pandoc ({exc})")
    try:
        import weasyprint  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local environment
        pytest.skip(f"Prérequis rapport manquant: weasyprint ({exc})")


def _build_user_payload(transcription: str, speaker_context: str | None) -> str:
    context = (speaker_context or "").strip()
    if not context:
        return transcription
    return f"Contexte sur les interlocuteurs :\n{context}\n\n{transcription}"


def _extract_chat_content(data: dict) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        content = first.get("text")
        if isinstance(content, str):
            return content.strip()
    raise RuntimeError(f"Réponse Ollama inattendue (keys={sorted(data.keys())})")


def _ollama_chat_complete(
    *,
    model: str,
    system: str,
    user_text: str,
    temperature: float = 0.1,
    timeout_s: int = 600,
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    openai_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "stream": False,
    }
    openai_resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=openai_payload,
        timeout=timeout_s,
    )
    if openai_resp.status_code == 200:
        return _extract_chat_content(openai_resp.json())

    # Fallback to Ollama native API for installations without /v1 compatibility.
    ollama_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }
    ollama_resp = requests.post(
        f"{base_url}/api/chat",
        json=ollama_payload,
        timeout=timeout_s,
    )
    if ollama_resp.status_code != 200:
        raise RuntimeError(
            "Ollama indisponible ou erreur de génération "
            f"(v1={openai_resp.status_code}, api={ollama_resp.status_code})"
        )
    data = ollama_resp.json()
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Réponse Ollama vide ou invalide.")
    return content.strip()


@pytest.mark.slow
def test_generate_two_reports_mistral_large_vs_ollama_gpt_oss_20b(tmp_path: Path) -> None:
    """
    Génère deux rapports sur la même transcription réelle:
    1) Mistral Large (flux standard du pipeline),
    2) Ollama local avec gpt-oss:20b.
    """
    _skip_if_missing_real_inputs()
    _skip_if_missing_report_prereqs()
    if not os.getenv("MISTRAL_API_KEY"):
        pytest.skip("MISTRAL_API_KEY absent: impossible d'appeler Mistral Large.")

    try:
        health = requests.get(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/tags",
            timeout=10,
        )
        if health.status_code != 200:
            pytest.skip(f"Ollama ne répond pas correctement (/api/tags -> {health.status_code}).")
    except requests.RequestException as exc:
        pytest.skip(f"Ollama indisponible en local: {exc}")

    transcription = REAL_TRANSCRIPTION_PATH.read_text(encoding="utf-8")
    meta = json.loads(REAL_META_PATH.read_text(encoding="utf-8"))

    meeting_date = str(meta.get("meeting_date") or "2025-11-01")
    prompt_key = str(meta.get("meeting_report_type") or "entretien_collaborateur")
    speaker_context = str(meta.get("speaker_context") or "").strip() or None

    prompts_path = PROJECT_ROOT / "src/asr_jetson/config/mistral_prompts.json"
    prompt = mistral_client.load_prompts(str(prompts_path), key=prompt_key)
    meeting_date_label = format_meeting_date_literal(meeting_date)
    user_text = prompt.user_prefix.format(meeting_date=meeting_date_label) + _build_user_payload(
        transcription, speaker_context
    )

    mistral_report = mistral_client.chat_complete(
        model=prompt.model,
        system=prompt.system,
        user_text=user_text,
        temperature=0.1,
    )

    ollama_report = _ollama_chat_complete(
        model="gpt-oss:20b",
        system=prompt.system,
        user_text=user_text,
        temperature=0.1,
    )

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = tmp_path / "json" / "no_anonymization_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text("{}", encoding="utf-8")

    mistral_anon_md = reports_dir / "meeting_report_mistral_large_anonymized.md"
    ollama_anon_md = reports_dir / "meeting_report_ollama_gpt-oss-20b_anonymized.md"
    mistral_anon_md.write_text(mistral_report, encoding="utf-8")
    ollama_anon_md.write_text(ollama_report, encoding="utf-8")

    mistral_outputs = generate_pdf_report(
        anonymized_markdown_path=mistral_anon_md,
        mapping_json_path=mapping_path,
        output_dir=tmp_path,
        run_id="mistral_large",
        title="Compte rendu Mistral Large",
        prompt_key=prompt_key,
        meeting_date=meeting_date,
        audio_stem="mistral_large",
        run_time="101010",
    )
    ollama_outputs = generate_pdf_report(
        anonymized_markdown_path=ollama_anon_md,
        mapping_json_path=mapping_path,
        output_dir=tmp_path,
        run_id="ollama_gpt-oss-20b",
        title="Compte rendu Ollama gpt-oss:20b",
        prompt_key=prompt_key,
        meeting_date=meeting_date,
        audio_stem="ollama_gpt-oss-20b",
        run_time="202020",
    )

    mistral_pdf = Path(mistral_outputs["report_pdf"])
    ollama_pdf = Path(ollama_outputs["report_pdf"])
    mistral_docx = Path(mistral_outputs["report_docx"])
    ollama_docx = Path(ollama_outputs["report_docx"])
    mistral_md = Path(mistral_outputs["report_markdown"])
    ollama_md = Path(ollama_outputs["report_markdown"])

    assert mistral_anon_md.exists()
    assert ollama_anon_md.exists()
    assert mistral_md.exists()
    assert ollama_md.exists()
    assert mistral_pdf.exists()
    assert ollama_pdf.exists()
    assert mistral_docx.exists()
    assert ollama_docx.exists()
    assert "mistral_large" in mistral_pdf.name
    assert "ollama_gpt-oss-20b" in ollama_pdf.name
    assert mistral_report.strip()
    assert ollama_report.strip()
    assert len(mistral_report) > 500
    assert len(ollama_report) > 500
    assert "###" in mistral_report
    assert "###" in ollama_report
