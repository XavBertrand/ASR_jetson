from __future__ import annotations
from pathlib import Path
from typing import Optional
import json

import asr_jetson.postprocessing.mistral_client as mistral_client
from asr_jetson.postprocessing.anonymizer import Anonymizer  # ta classe existante
from asr_jetson.postprocessing.docx_export import save_docx_from_markdown_sections

def generate_meeting_report(
    anonymized_txt_path: Path,
    mapping_json_path: Path,
    prompts_json_path: Path,
    prompt_key: str = "meeting_analysis",
    out_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
):
    """
    1) Lit la transcription anonymisée (txt)
    2) Lance Mistral Large avec le prompt JSON (key configurable)
    3) Désanonymise le rapport LLM via mapping
    4) Exporte TXT (anon + désanon) et DOCX
    """
    out_dir = out_dir or anonymized_txt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) charge prompt + texte
    prompt = mistral_client.load_prompts(str(prompts_json_path), key=prompt_key)
    anon_text = Path(anonymized_txt_path).read_text(encoding="utf-8")

    # 2) Mistral Large
    user_payload = prompt.user_prefix + anon_text
    analysis_anonymized = mistral_client.chat_complete(model=prompt.model, system=prompt.system, user_text=user_payload)

    # 3) Désanonymisation
    mapping = json.loads(Path(mapping_json_path).read_text(encoding="utf-8"))
    analysis_deanonymized = Anonymizer.deanonymize(analysis_anonymized, mapping, restore="canonical")

    # 4) Sauvegardes
    base = (run_id or Path(anonymized_txt_path).stem.replace("_anon_clean", ""))
    out_dir = Path(out_dir)
    if out_dir.name in {"txt", "reports", "docx"} and out_dir.parent != out_dir:
        outputs_root = out_dir.parent
    elif out_dir.name == "outputs":
        outputs_root = out_dir
    else:
        outputs_root = out_dir
    outputs_root.mkdir(parents=True, exist_ok=True)
    reports_dir = outputs_root / "reports"  # text exports
    docx_dir = outputs_root / "docx"        # docx exports
    reports_dir.mkdir(parents=True, exist_ok=True)
    docx_dir.mkdir(parents=True, exist_ok=True)

    out_txt_anon = reports_dir / f"{base}_meeting_report_anonymized.txt"
    out_txt = reports_dir / f"{base}_meeting_report.txt"
    out_docx = docx_dir / f"{base}_meeting_report.docx"

    out_txt_anon.write_text(analysis_anonymized, encoding="utf-8")
    out_txt.write_text(analysis_deanonymized, encoding="utf-8")
    save_docx_from_markdown_sections(analysis_deanonymized, str(out_docx), title=f"Compte rendu — {base}")

    return {
        "report_anonymized_txt": str(out_txt_anon),
        "report_txt": str(out_txt),
        "report_docx": str(out_docx),
    }
