import json
import os
from pathlib import Path

import pytest

import asr_jetson.postprocessing.anonymizer as anonymizer_mod
from asr_jetson.postprocessing.anonymizer import (
    Settings,
    anonymize_text,
    deanonymize_text,
)

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore


@pytest.fixture()
def sample_text(tmp_path: Path) -> str:
    content = (
        "Compte rendu du 12/09/2025 — Séance plénière du comité stratégique.\n\n"
        "Contact : xavier@example.com, tél : +33 6 12 34 56 78.\n\n"
        "1) Ouverture de séance\n"
        "À 08h45 précises, Xavier Bertrand accueille Alice Dupont et rappelle les décisions actées lors du précédent atelier.\n"
        "Quelques instants plus tard, Maître Elise Lemaire (cabinet Action Avocats) confirme que l'accord-cadre signé avec le "
        "consortium Zenith Labs reste valide, malgré l'ajout de nouvelles clauses RGPD.\n\n"
        "2) Points budgétaires\n"
        "L'équipe finance, représentée par Paul-André Martin, annonce une révision du budget: +12% sur le poste « infrastructure ».\n"
        "Une discussion s'engage avec le directeur des opérations, Karim Haddad, afin d'évaluer l'impact sur les délais de livraison.\n"
        "Le groupe approuve finalement la proposition, sous réserve d'une note de synthèse adressée à conseil@zenith-labs.fr avant le 20/09.\n\n"
        "3) Plan d'actions\n"
        "Les participants conviennent de mobiliser trois chantiers:\n"
        "  - Migration des données sensibles vers le cluster sécurisé de Toulouse.\n"
        "  - Mise à jour des procédures d'escalade incidents, sous la responsabilité de Claire Rousseau.\n"
        "  - Organisation d'une session de sensibilisation juridique pour les équipes produit, animée par Action Avocats.\n"
        "Jean-Luc Morel rappelle que toute communication externe doit transiter par communication@zenith-labs.fr.\n\n"
        "4) Clôture\n"
        "Avant de lever la séance, Xavier informe qu'une réunion de suivi se tiendra à Lyon le 30/09 à 10h00.\n"
        "Il demande également que l'on prépare une synthèse à destination du siège parisien et que Laura Benchetrit vérifie la conformité des contrats annexes.\n"
    )
    path = tmp_path / "transcript.txt"
    path.write_text(content, encoding="utf-8")
    return content


@pytest.fixture(scope="module")
def ollama_settings():
    if anonymizer_mod.AutoTokenizer is None or anonymizer_mod.AutoModelForTokenClassification is None:
        pytest.skip("transformers is not installed; real NER is unavailable.")
    if requests is None:
        pytest.skip("requests package missing; cannot probe Ollama service.")

    base_url = anonymizer_mod.normalize_ollama_base_url(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model_name = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")

    probe_url = f"{base_url}/api/version"
    try:
        health = requests.get(probe_url, timeout=5)
    except requests.RequestException as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Ollama service not reachable at {base_url}: {exc}")
    else:
        if health.status_code >= 500:
            pytest.skip(f"Ollama service unhealthy at {probe_url}: HTTP {health.status_code}")

    return Settings(
        model_id="Jean-Baptiste/camembert-ner",
        device=-1,
        enable_llm_qc=True,
        ollama_base_url=base_url,
        llm_model=model_name,
        ollama_timeout=20,
    )


@pytest.mark.slow
def test_anonymize_and_restore_roundtrip(sample_text, ollama_settings):
    anon_text, mapping = anonymize_text(
        sample_text,
        settings=ollama_settings,
    )

    assert "<PER_" in anon_text
    assert "<ORG_" in anon_text
    assert "<EMAIL_" in anon_text
    assert "<PHONE_" in anon_text
    assert "<LOC_" in anon_text

    entities = {entity["tag"]: entity for entity in mapping["entities"]}
    assert any(entity["type"] == "PERSON" for entity in entities.values())
    assert mapping["summary"]["PERSON"] >= 1

    restored = deanonymize_text(anon_text, mapping, restore="canonical")
    assert "Xavier Bertrand" in restored
    assert "Action Avocats" in restored
    assert "xavier@example.com" in restored
    assert "+33 6 12 34 56 78" in restored


@pytest.mark.slow
def test_mapping_serialization(tmp_path, sample_text, ollama_settings):
    anon_text, mapping = anonymize_text(
        sample_text,
        settings=ollama_settings,
    )

    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    reloaded = json.loads(mapping_path.read_text(encoding="utf-8"))

    assert reloaded["entities"]
    restored = deanonymize_text(anon_text, reloaded, restore="longest")
    assert "Xavier Bertrand" in restored


@pytest.mark.slow
def test_anonymizer_metadata_contains_ollama_info(sample_text, ollama_settings):
    anon_text, mapping = anonymize_text(sample_text, settings=ollama_settings)

    assert "<PER_" in anon_text
    assert mapping["summary"]["PERSON"] >= 1
    assert mapping["meta"]["ollama_base_url"] == ollama_settings.ollama_base_url
    assert mapping["meta"]["llm_model"] == ollama_settings.llm_model
    assert mapping["meta"]["enable_llm_qc"] is True


@pytest.mark.slow
def test_long_text_chunked_roundtrip_with_llm_validation(ollama_settings):
    long_text = (
        "Récapitulatif confidentiel — mission Horizon 2030.\n"
        "Le 05/04/2026 à 08h20, Jeanne Moreau et Adrien Leclerc se retrouvent au siège de NovaLyra "
        "situé au 14 rue des Peupliers, 75012 Paris, pour finaliser le plan de déploiement.\n"
        "L'équipe projet rejoint la Salle Atlas à 09h00 : Sophie Bernard (produit), Mehdi Rahmani (sécurité), "
        "ainsi que l'auditrice externe, Maître Ingrid Keller du cabinet Action Avocats.\n"
        "Les contacts opérationnels sont joignables via operations@novalyra.eu ; les urgences logistiques "
        "peuvent être signalées au +33 6 98 76 54 32 ou au standard 01 44 09 87 65.\n"
        "La feuille de route prévoit trois vagues de mise en service : Toulouse en septembre, "
        "Lyon en octobre, et Lille en novembre. Chaque venue doit être confirmée 72 heures à l'avance.\n"
        "Une session de tests de charge est planifiée le 18/05/2026 pour valider la résilience du système. "
        "En cas d'incident, contacter Paul Giraud au +33 7 45 23 19 88 ou par courriel à paul.giraud@novalyra.eu.\n"
        "Clôture de la réunion à 18h15, avec demande d'un compte rendu détaillé à remettre à Conseil NovaLyra avant le 10/04.\n"
    )

    anon_text, mapping = anonymize_text(
        long_text,
        settings=ollama_settings,
        max_block_chars=600,
        max_block_sents=4,
    )

    assert "<PER_" in anon_text
    assert "<ORG_" in anon_text
    assert "<EMAIL_" in anon_text
    assert "<PHONE_" in anon_text
    assert mapping["summary"]["PERSON"] >= 2
    assert mapping["summary"]["EMAIL"] >= 1
    assert mapping["summary"]["PHONE"] >= 2

    restored = deanonymize_text(anon_text, mapping, restore="canonical")

    assert "Jeanne Moreau" in restored
    assert "Action Avocats" in restored
    assert "operations@novalyra.eu" in restored
    assert "18/05/2026" in restored
