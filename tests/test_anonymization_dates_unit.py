from __future__ import annotations

import os
from pathlib import Path
import re

import pytest

from asr_jetson.postprocessing import mistral_client
from asr_jetson.postprocessing.transformer_anonymizer import TransformerAnonymizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _has_gliner() -> bool:
    try:  # pragma: no cover - optional dependency check
        import gliner  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _has_mistral() -> bool:
    if not os.environ.get("MISTRAL_API_KEY"):
        return False
    try:  # pragma: no cover - optional dependency check
        import mistralai  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def test_anonymization_preserves_dates_for_mistral_report() -> None:
    if not _has_gliner():
        pytest.skip("GLiNER non disponible pour ce test d'intégration.")
    if not _has_mistral():
        pytest.skip("Mistral API indisponible (MISTRAL_API_KEY ou package manquant).")

    transcription = (
        "SPEAKER_1 : Bonjour, je suis Claire Martin du cabinet Axis Legal, je reprends le dossier.\n"
        "SPEAKER_2 : Je suis Jean Dupont. L'accident du 4 février 2023 a été déclaré, "
        "et j'ai informé mon employeur dans la foulée.\n"
        "SPEAKER_1 : D'accord. Pour poser le contexte : le licenciement a eu lieu le 15/09/2023, "
        "et l'audience est fixée au 18 avril 2024.\n"
        "SPEAKER_2 : Nous avons reçu la convocation le 2024-03-05 et devons répondre sous 15 jours. "
        "J'ai déjà transmis une première note interne.\n"
        "SPEAKER_1 : La médiation du 12/06/2024 a été annulée, avec une expertise attendue d'ici 3 semaines. "
        "Il faudra aussi récupérer le dossier médical complet.\n"
        "SPEAKER_2 : Sur les échanges précédents, l'assureur a parlé d'une franchise, "
        "mais je n'ai pas retrouvé le montant exact. Il y a eu deux relances par mail.\n"
        "SPEAKER_1 : Je note. L'objectif est clair : préparer les pièces, clarifier la chronologie, "
        "et sécuriser les éléments factuels avant l'audience.\n"
        "SPEAKER_2 : De mon côté, je peux fournir les attestations de collègues et le certificat initial.\n"
        "SPEAKER_1 : Parfait. On reste factuel. On rappelle aussi que certains délais internes "
        "sont courts et qu'on ne doit pas rater les échéances.\n"
        "SPEAKER_2 : Oui, et je veux éviter une mauvaise interprétation des dates, "
        "car le récit dépend vraiment de ces repères.\n"
        "SPEAKER_1 : Très bien, on s'aligne là-dessus et on priorise les documents clés.\n"
    )
    expected_dates = [
        "4 février 2023",
        "15/09/2023",
        "18 avril 2024",
        "2024-03-05",
        "12/06/2024",
    ]
    expected_durations = ["15 jours", "3 semaines"]

    domain_entities = {
        "PERSON": ["Claire Martin", "Jean Dupont"],
        "ORGANIZATION": ["Axis Legal"],
    }

    anonymizer = TransformerAnonymizer(
        model_name="urchade/gliner_multi_pii-v1",
        preserve_dates=True,
        domain_entities=domain_entities,
        device="cpu",
    )
    anonymized_text, mapping = anonymizer.anonymize_with_tags(transcription)

    for value in expected_dates + expected_durations:
        assert value in anonymized_text, f"Le repère temporel doit rester visible: {value}"

    for raw in ["Claire Martin", "Jean Dupont", "Axis Legal"]:
        assert raw not in anonymized_text, f"PII non anonymisée: {raw}"

    assert all(info.get("label") != "DATE" for info in mapping.get("entities", {}).values())

    prompts_path = PROJECT_ROOT / "src/asr_jetson/config/mistral_prompts.json"
    prompt = mistral_client.load_prompts(str(prompts_path))
    report = mistral_client.chat_complete(
        prompt.model,
        prompt.system,
        prompt.user_prefix.format(meeting_date="2024-05-10") + anonymized_text,
        temperature=0.0,
    )

    def _assert_any_match(text: str, patterns: list[str], label: str) -> None:
        if not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            joined = " | ".join(patterns)
            raise AssertionError(f"Le CR doit restituer {label} (aucun match: {joined})")

    date_matchers = {
        "4 février 2023": [
            r"\b4\s+f(?:é|e)vrier\s+2023\b",
            r"\b04[/\-.]02[/\-.]2023\b",
            r"\b2023[/\-.]02[/\-.]04\b",
        ],
        "15/09/2023": [
            r"\b15[/\-.]09[/\-.]2023\b",
            r"\b15\s+sept(?:embre|\.)?\s+2023\b",
        ],
        "18 avril 2024": [
            r"\b18\s+avril\s+2024\b",
            r"\b18[/\-.]04[/\-.]2024\b",
            r"\b2024[/\-.]04[/\-.]18\b",
        ],
        "2024-03-05": [
            r"\b2024[/\-.]03[/\-.]05\b",
            r"\b0?5\s+mars\s+2024\b",
        ],
        "12/06/2024": [
            r"\b12[/\-.]06[/\-.]2024\b",
            r"\b12\s+juin\s+2024\b",
            r"\b2024[/\-.]06[/\-.]12\b",
        ],
    }
    duration_matchers = {
        "15 jours": [r"\b15\s+jours\b", r"\bquinze\s+jours\b"],
        "3 semaines": [r"\b3\s+semaines\b", r"\btrois\s+semaines\b"],
    }

    for label, patterns in date_matchers.items():
        _assert_any_match(report, patterns, label)
    for label, patterns in duration_matchers.items():
        _assert_any_match(report, patterns, label)
