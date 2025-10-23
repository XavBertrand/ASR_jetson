# tests/test_meeting_report.py
from __future__ import annotations
import json
from pathlib import Path

import pytest

# On importe la fonction orchestratrice
# (le chemin ci-dessous suit la structure que je t'ai proposée)
from src.asr_jetson.postprocessing.meeting_report import generate_meeting_report

# On importera les modules à monkeypatcher via leurs chemins réels
import src.asr_jetson.postprocessing.mistral_client as mistral_client_mod
import src.asr_jetson.postprocessing.meeting_report as meeting_report_mod

from docx import Document


@pytest.fixture
def temp_project(tmp_path: Path):
    """
    Fabrique un mini "projet" temporaire avec :
      - un transcript anonymisé,
      - un mapping d'anonymisation,
      - un fichier de prompts JSON.
    Retourne un dict avec les chemins utiles.
    """
    root = tmp_path
    (root / "config").mkdir(parents=True, exist_ok=True)
    outputs_txt_dir = root / "outputs" / "txt"
    outputs_txt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Transcription anonymisée nettoyée (out_txt_anon_clean)
    anon_txt = outputs_txt_dir / "testfile_anon_clean.txt"
    anon_txt.write_text(
        "Bonjour <PER_1>. Nous avons évoqué le dossier <ORG_1> et fixé une échéance au 12/11.\n",
        encoding="utf-8",
    )

    # 2) Mapping anonymisation
    mapping_json = outputs_txt_dir / "testfile_anon_mapping.json"
    mapping = {
        "entities": [
            {
                "tag": "<PER_1>",
                "type": "PERSON",
                "canonical": "Delphine",
                "mentions": ["Delphine"],
                "sources": ["stub"],
            },
            {
                "tag": "<ORG_1>",
                "type": "ORGANIZATION",
                "canonical": "UDAF",
                "mentions": ["UDAF"],
                "sources": ["stub"],
            },
        ],
        "summary": {"PERSON": 1, "ORGANIZATION": 1},
        "tag_lookup": {"<PER_1>": "Delphine", "<ORG_1>": "UDAF"},
        "meta": {"model_id": "stub"},
    }
    mapping_json.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Prompts JSON (clé meeting_analysis)
    prompts_json = root / "config" / "mistral_prompts.json"
    prompts_json.write_text(
        json.dumps({
            "meeting_analysis": {
                "model": "mistral-large-latest",
                "system": "Tu es un assistant d'analyse de réunions professionnelles.",
                "user_prefix": "TRANSCRIPTION :\n"
            }
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "root": root,
        "anon_txt": anon_txt,
        "mapping_json": mapping_json,
        "prompts_json": prompts_json,
        "outputs_txt_dir": outputs_txt_dir,
    }


def _fake_llm_output():
    return (
        "### RÉSUMÉ EXÉCUTIF\n"
        "Point sur le dossier <ORG_1>. Décision d'avancer l'échéance.\n\n"
        "### PARTICIPANTS\n"
        "- <PER_1> / Avocat(e) / Féminin\n\n"
        "### SUJETS ABORDÉS\n"
        "- Dossier — résumé: échéance au 12/11 «Nous devons finaliser le dossier». \n\n"
        "### DÉCISIONS\n"
        "- Avancer l’échéance au 12/11 — prise par <PER_1>\n\n"
        "### ACTIONS\n"
        "- Envoyer les pièces — Responsable: <PER_1> — Échéance: 12/11\n\n"
        "### PROCHAINES ÉTAPES\n"
        "- Réunion de suivi — Responsable: <PER_1> — Délai: 1 semaine\n"
    )



def test_generate_meeting_report_end_to_end_no_network(
    temp_project, monkeypatch
):
    """
    Test end-to-end du rapport :
      - Mock de l'appel LLM (aucun réseau)
      - Mock de la désanonymisation via deanonymize_text()
      - Vérification des 3 sorties (txt anonymisé, txt désanonymisé, docx)
    """

    # === 1) Mock LLM : intercepter chat_complete et renvoyer une réponse prédictible ===
    def fake_chat_complete(model: str, system: str, user_text: str) -> str:
        # On pourrait vérifier ici que user_text contient la transcription anonymisée
        assert "TRANSCRIPTION" in user_text  # vient de user_prefix du prompt
        assert "<PER_1>" in user_text or "<ORG_1>" in user_text
        return _fake_llm_output()

    # Patch ce que meeting_report.py utilise réellement :
    monkeypatch.setattr(meeting_report_mod.mistral_client, "chat_complete", fake_chat_complete)

    # === 2) Mock désanonymisation : deanonymize_text ===
    # On remplace la fonction par une version simple qui applique le mapping fourni.
    def fake_deanon(text: str, mapping: dict, restore: str = "canonical") -> str:

        lookup = dict(mapping.get("tag_lookup", {}))
        if not lookup:
            for entity in mapping.get("entities", []):
                lookup[entity["tag"]] = entity.get("canonical", entity["tag"])
        out = text
        for tag, real in lookup.items():
            out = out.replace(tag, real)
        return out

    monkeypatch.setattr(meeting_report_mod, "deanonymize_text", fake_deanon)

    # === 3) Exécution ===
    root = temp_project["root"]
    anon_txt = temp_project["anon_txt"]
    mapping_json = temp_project["mapping_json"]
    prompts_json = temp_project["prompts_json"]
    outputs_txt_dir = temp_project["outputs_txt_dir"]

    result = generate_meeting_report(
        anonymized_txt_path=anon_txt,
        mapping_json_path=mapping_json,
        prompts_json_path=prompts_json,
        prompt_key="meeting_analysis",
        out_dir=outputs_txt_dir,       # on sort dans outputs/txt
        run_id="testfile",
    )

    # === 4) Vérifications des chemins retournés ===
    assert "report_anonymized_txt" in result
    assert "report_txt" in result
    assert "report_docx" in result

    p_anon = Path(result["report_anonymized_txt"])
    p_txt = Path(result["report_txt"])
    p_docx = Path(result["report_docx"])

    assert p_anon.exists(), "Le .txt anonymisé du rapport doit exister"
    assert p_txt.exists(), "Le .txt désanonymisé du rapport doit exister"
    assert p_docx.exists(), "Le .docx du rapport doit exister"

    # === 5) Contenu des TXT ===
    anon_text = p_anon.read_text(encoding="utf-8")
    deanon_text = p_txt.read_text(encoding="utf-8")

    # La version anonymisée contient encore des tags
    assert "<PER_1>" in anon_text and "<ORG_1>" in anon_text

    # La version désanonymisée remplace bien par les vrais noms
    assert "Delphine" in deanon_text
    assert "UDAF" in deanon_text
    assert "<PER_1>" not in deanon_text
    assert "<ORG_1>" not in deanon_text

    # Les sections ### sont bien présentes (utile pour le formatage docx)
    for section in [
        "### RÉSUMÉ EXÉCUTIF",
        "### PARTICIPANTS",
        "### SUJETS ABORDÉS",
        "### DÉCISIONS",
        "### ACTIONS",
        "### PROCHAINES ÉTAPES",
    ]:
        assert section in deanon_text

    # === 6) Lecture et checks rapides du DOCX ===
    doc = Document(str(p_docx))
    full_doc_text = "\n".join([p.text for p in doc.paragraphs])

    # Doit contenir les mêmes informations clé (désanonymisées)
    assert "Delphine" in full_doc_text
    assert "UDAF" in full_doc_text
    # Et au moins un titre de section transformé en heading
    assert any("RÉSUMÉ EXÉCUTIF" in p.text for p in doc.paragraphs)
