# -*- coding: utf-8 -*-
import json
import re
import os
import pytest
from pathlib import Path

from asr_jetson.postprocessing.anonymizer import Anonymizer

pytestmark = pytest.mark.slow  # on marque tout le fichier comme "slow"

@pytest.fixture(scope="session")
def hf_env():
    # Conseillé pour éviter les re-downloads en CI/Jetson
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    # Limite mémoire tokenizer si besoin
    return True

@pytest.fixture
def catalog_path(tmp_path: Path):
    data = [
        {"pattern": "Action Avocats", "label": "ORG"},
        {"pattern": "Airbus", "label": "ORG"},
        # aliases à traiter comme PERSON si anon_catalog_as_person=True
        {"pattern": "Xavier", "label": "CAT"},
        {"pattern": "XAVIER BERTRAND", "label": "CAT"},
        {"pattern": "M. Bertrand", "label": "CAT"},
        {"pattern": "Xav", "label": "CAT"},
    ]
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

@pytest.fixture(scope="session")
def anonymizer_gpu(hf_env):
    # device="cuda" utilisera le GPU si dispo; sinon retombe CPU
    # Sur Jetson Orin Nano, garde batch=1 (pipeline le fait déjà)
    return Anonymizer(model_name="cmarkea/distilcamembert-base-ner", device="auto")

def make_long_text(min_words: int = 2200) -> str:
    """
    Construit un texte très long (> min_words mots) en dupliquant un bloc
    qui contient des alias, emails, téléphones, ORG/LIEU, etc.
    On varie les emails/téléphones pour éviter les faux positifs de cache.
    """
    seed = (
        "Section 0 — Contexte. Xavier échange avec M. Bertrand. "
        "Puis XAVIER BERTRAND répond à Xav chez Action Avocats. "
        "Contact: xavier.bertrand@example.com, tel +33 6 12 34 56 78. "
        "On mentionne aussi Toulouse et Paris. "
    )
    parts = [seed]
    n = 1
    while len(" ".join(parts).split()) < min_words:
        parts.append(
            f"\n\nSection {n} — Suite. "
            f"Chez Action Avocats, M Bertrand et Xav valident. "
            f"Email: xavier.bertrand+seg{n}@example.org ; "
            f"Tél: 06-11-22-33-{40+n:02d}. "
            f"Rappel: AIRBUS et FD&S présents ; Voxtral/Whisper en projet. "
            f"A Paris puis à Toulouse. "
        )
        # Ajoute quelques variantes avec titres et prénoms pour tester alias
        parts.append(
            f"Par ailleurs, Monsieur Bertrand confirme ; "
            f"m Bertrand (var) acquiesce ; "
            f"XAVIER BERTRAND précise un point ; "
            f"Xav envoie un SMS ; Mme Martin est notée. "
        )
        # Une phrase où ORG arrive tard dans le document
        parts.append(
            f"En fin de segment {n}, nouvelle mention tardive : Action Avocats (tardif). "
        )
        n += 1
    long_text = "".join(parts)
    return long_text


@pytest.mark.parametrize(
    "text",
    [
        # Variantes alias/casse/titre
        "Xavier a vu M. Bertrand. Puis XAVIER BERTRAND a répondu à Xav chez Action Avocats.",

        # Fautes légères + email/tel + ORG catalogue
        "M Bertrand a ecrit à Xav via xavier.bertrand@example.com et a appelé le +33 6 12 34 56 78 chez Action Avocats.",

        # Plusieurs personnes sans alias (contrôle des balises isolées)
        " puis Guilhem a discuté avec Candice chez Delphine Heinrich avant de se rendre chez Nicole et Gérard.",

        # Bruit & typos : ‘Axavier’ (typo), ‘Bertrand,’ avec ponctuation
        "Hier, Axavier a croisé Bertrand, puis XAVIER BERTRAND… plus tard chez Action Avocats.",

        # Cas complexe multi-paragraphes pour test complet
        (
            "Réunion du 12/09/2025 à 09:30 – Compte rendu préliminaire.\n"
            "\n"
            "1) Participants & contexte\n"
            "Xavier a ouvert la séance. M Bertrand a rappelé les objectifs initiaux. "
            "Un peu plus tard, XAVIER BERTRAND a confirmé que « le périmètre reste identique ». "
            "Dans le couloir, Xav a croisé Mme Martin. Puis Monsieur  Dupont (sans accent volontaire) s'est excusé pour son retard. "
            "À noter : Delphine-Heinrich a été mentionnée par Candice, et GÉRARD était en distanciel depuis Toulouse.\n"
            "\n"
            "2) Coordonnées, canaux & identifiants\n"
            "Pour toute question, écrire à xavier.bertrand@example.com, ou à xav.bertrand+legal@sub.example.org; "
            "au besoin appeler le +33 6 12 34 56 78 ou bien 06-11-22-33-44 (poste 19). "
            "Le numéro de standard d'Action Avocats reste le 01 23 45 67 89. "
            "Adresse temporaire du bureau: 12, rue de l'Exemple 31000 Toulouse. "
            "Ancien IBAN (à ne pas diffuser): FR76 3000 6000 0112 3456 7890 189. "
            "SIREN pour test: 123 456 789.\n"
            "\n"
            "3) Organisations, projets & lieux\n"
            "Action Avocats a proposé une réunion conjointe avec AIRBUS et FD&S. "
            "Le sous-projet CARM avance; Voxtral et Whisper seront intégrés. "
            "On a aussi parlé d'AeroNEF et de TensorRT-LLM. "
            "La prochaine séance pourrait se tenir chez Action  Avocat (typo volontaire) à Paris, ou chez Action Avocats-Toulouse. "
            "Un déplacement à Saint-Étienne est envisagé; parfois écrit St. Etienne ou St-Étienne.\n"
            "\n"
            "4) Fragments, collages et cas tordus (espaces et ponctuation)\n"
            "…puis<M Bertrand> a répondu: «OK». Ensuite,il a ajouté qu'il verrait avec<Xav>. "
            "Candicea dit que Delphine  Heinrich validera;et Gérard,peut-être. "
            "Chez<Mme Martin> on a noté un souci; et<Xavier> a promis un retour. "
            "Le document final sera envoyé à xavier.bertrand@example.com, ainsi qu'à  dupont.monsieur@example.net . "
            "Note: 'Axavier' apparaît parfois comme coquille pour Xavier; idem ‘Bertrand,’ avec virgule collée.\n"
            "\n"
            "5) Variantes de titres & minuscules/majuscules\n"
            "M.  Bertrand a confirmé le budget. MR BERTRAND (ancienne orthographe) l'a répété. "
            "Monsieur   Bertrand a approuvé la feuille de route. "
            "m Bertrand (minuscule volontaire) a conclu.\n"
            "\n"
            "6) Résumé opérationnel\n"
            "— Personnes à suivre: Xavier / XAVIER BERTRAND / M Bertrand / Xav ; "
            "Mme Martin ; Candice ; Delphine-Heinrich ; Gérard ; Monsieur Dupont. "
            "— Orgs: Action Avocats / AIRBUS / FD&S / Action  Avocat (variante) / Action Avocats-Toulouse. "
            "— Lieux: Paris / Toulouse / Saint-Étienne (St. Etienne, St-Étienne). "
            "— Canaux: emails ci-dessus ; téléphones +33 6 12 34 56 78, 06-11-22-33-44, 01 23 45 67 89.\n"
        ),

        # 🚀 Très long texte (>2048 tokens estimés) pour valider le chunking
        make_long_text(min_words=2200),
    ],
)

def test_dynamic_anonymization_with_real_model(text, anonymizer_gpu, catalog_path, tmp_path):
    anon_text, mapping = anonymizer_gpu.anonymize(
        text,
        catalog_path=str(catalog_path),
        catalog_label_default="CAT",
        catalog_fuzzy_threshold=88,     # un peu plus tolérant pour tester le fuzzy
        catalog_as_person=True,
    )

    # --- 1) Toutes les mentions Person doivent être taguées <NOM_k> ---
    nom_tags = re.findall(r"<NOM_(\d+)>", anon_text)
    assert len(nom_tags) >= 1, "Au moins une balise <NOM_k> attendue"
    has_title_plus_name = bool(
        re.search(r"\b(M|Mr|Monsieur|Mme|Madame|Mlle|Mademoiselle)\.?\s+[A-ZÉÈÀÂÎÔÙÜÇ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", text))
    has_email_localpart = bool(re.search(r"\b[\w.+-]+@", text))
    # Heuristique simple de répétition nominale
    caps = re.findall(r"\b[A-ZÉÈÀÂÎÔÙÜÇ][a-zà-öø-ÿ'’-]+\b", text)
    name_repeat = len(caps) != len(set(caps))

    requires_alias_cluster = has_title_plus_name or has_email_localpart or name_repeat

    if requires_alias_cluster:
        assert any(nom_tags.count(k) >= 2 for k in set(nom_tags)), \
            "Alias attendus non regroupés en <NOM_k>"

    # --- 2) ORG (Action Avocats) doit être balisé ---
    if "Action Avocats" in text:
        assert "<ORG_" in anon_text, "Une balise <ORG_k> est attendue (Action Avocats)"

    # --- 3) Emails & Tel masqués si présents ---
    if "@" in text:
        assert "<EMAIL_" in anon_text, "Email non balisé"
    if any(c.isdigit() for c in text):
        # num tel approximatif → tag <TEL_k> attendu si motif détecté
        assert "<TEL_" in anon_text or "+33" not in text, \
            "Téléphone non balisé (si motif standard)"

    # --- 4) Mapping réversible cohérent ---
    # Le canonical d'une personne devrait être la forme la plus descriptive (souvent la plus longue)
    for tag, info in mapping.items():
        if tag.startswith("<NOM_"):
            assert info["type"] == "PERSON"
            assert len(info.get("mentions", [])) >= 1
            # canonical parmi les mentions les plus longues
            assert info["canonical"] in info["mentions"] or len(info["canonical"]) >= max(len(m) for m in info["mentions"])

    # --- 5) Dé-anonymisation ---
    dean = Anonymizer.deanonymize(anon_text, mapping, restore="canonical")
    # On doit retrouver les marqueurs clés (au moins ORG si présent dans le texte)
    if "Action Avocats" in text:
        assert "Action Avocats" in dean

    # --- 6) Cas très long : vérifier que des balises apparaissent aussi en seconde moitié
    if len(text.split()) > 1500:
        mid = len(anon_text) // 2
        tail = anon_text[mid:]
        assert any(tok in tail for tok in ("<NOM_", "<ORG_", "<EMAIL_", "<TEL_>")), \
            "Aucune balise trouvée dans la seconde moitié — probable troncature NER/chunking"
        # et jusque tout à la fin
        tail_end = anon_text[-1000:]
        assert any(tok in tail_end for tok in ("<NOM_", "<ORG_", "<EMAIL_", "<TEL_>")), \
            "Aucune balise dans le dernier segment — chunking/offsets globaux à revoir"


    # Écritures d’artefacts (facultatif)
    (tmp_path / "anon.txt").write_text(anon_text, encoding="utf-8")
    (tmp_path / "map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
