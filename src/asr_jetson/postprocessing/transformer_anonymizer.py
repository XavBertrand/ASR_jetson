# src/asr_jetson/postprocessing/transformer_anonymizer.py

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from transformers import pipeline
try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _resolve_device_index(device: Optional[int | str]) -> int:
    """
    Convert a device hint into a transformers-compatible device index.
    """
    if isinstance(device, int):
        return device
    if device is None:
        return -1

    hint = str(device).strip().lower()
    if hint in {"auto", "cuda", "gpu"}:
        return 0 if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available() else -1
    if hint.startswith("cuda:"):
        try:
            idx = int(hint.split(":", 1)[1])
        except ValueError:
            idx = 0
        available = torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()
        return idx if available else -1
    if hint in {"cpu", "-1"}:
        return -1
    try:
        idx = int(hint)
        available = torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()
        return idx if available else -1
    except ValueError:
        return -1


class TransformerAnonymizer:
    """
    Anonymiseur basé sur CamemBERT NER - remplace Presidio pour le français.
    Performances bien meilleures sur transcriptions ASR françaises.
    """

    def __init__(
        self,
        model_name: str = "Jean-Baptiste/camembert-ner",
        whitelist: List[str] | None = None,
        domain_entities: Dict[str, List[str]] | None = None,
        device: int | str | None = "cuda",
    ):
        """
        Args:
            model_name: Modèle HuggingFace  utiliser
                       Recommandations par ordre :
                       1. "almanach/camembertav2-base" (SOTA 2024, 93.4% F1)
                       2. "Jean-Baptiste/camembert-ner" (excellent ASR/chat)
                       3. "cmarkea/distilcamembert-base-ner" (léger, rapide)
            whitelist: Mots à ne jamais anonymiser
            domain_entities: Dictionnaire {"PERSON": [...], "ORG": [...]}
                           d'entités connues de ton domaine
            device: Index ou hint ("cuda", "cpu", "auto") pour l'exécution du modèle
        """
        self.device_index = _resolve_device_index(device)
        print(f"🔄 Chargement du modèle {model_name}...")
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=self.device_index
        )
        print("✅ Modèle chargé")

        # ⚠️ WHITELIST : UNIQUEMENT des mots génériques, JAMAIS d'entités !
        base_whitelist = [
            "cabinet", "télétravail", "donc", "voilà",
            "ok", "oui", "non", "c'est", "c'était",
            "sms", "whatsapp", "mail", "internet", "email",
            "objectif", "dossier", "dossiers", "facture", "factures",
            "poste", "envie", "step", "univers", "conseil", "tresor", "trésor",
            "ouh", "euh", "bah"
        ]
        self.whitelist = set(s.lower() for s in (whitelist or []) + base_whitelist)

        # Entités du domaine (noms connus dans tes transcriptions)
        self.domain_entities = domain_entities or {}
        self._domain_lookup = self._build_domain_lookup(self.domain_entities)
        self._domain_values = {value.lower() for values in self.domain_entities.values() for value in values}

        # Seuils
        self.min_score = 0.5
        self.min_length = 2

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """
        Détecte les entités dans le texte.
        Gère les textes longs en les découpant par chunks avec overlap.
        """
        entities = []

        # 1️⃣ Détection par ML avec gestion des textes longs
        try:
            entities_ml = self._analyze_with_chunking(text)
            entities.extend(entities_ml)
        except Exception as e:
            print(f"⚠️ Erreur ML: {e}")

        # 2️⃣ Détection par dictionnaire domaine
        domain_ents = self._find_domain_entities(text)
        entities.extend(domain_ents)

        # 3️⃣ Patterns contextuels spécifiques ASR
        pattern_ents = self._find_contextual_patterns(text)
        entities.extend(pattern_ents)

        # Déduplique et filtre
        entities = self._deduplicate_entities(entities, text)

        return entities

    def _analyze_with_chunking(self, text: str, max_length: int = 400) -> List[Dict[str, Any]]:
        """
        Découpe le texte en chunks pour gérer les textes longs.
        CamemBERT a une limite de 512 tokens, on utilise 400 pour avoir de la marge.
        """
        entities = []

        # Si le texte est court, traite directement
        if len(text) < max_length:
            ml_results = self.ner_pipeline(text)
            for ent in ml_results:
                if ent["score"] >= self.min_score:
                    entities.append({
                        "start": ent["start"],
                        "end": ent["end"],
                        "entity_type": self._normalize_type(ent["entity_group"]),
                        "score": ent["score"],
                        "source": "ml"
                    })
            return entities

        # Sinon, découpe intelligemment (sur des phrases si possible)
        chunks = self._split_text_smart(text, max_length)

        current_offset = 0
        for chunk_text in chunks:
            if not chunk_text.strip():
                current_offset += len(chunk_text)
                continue

            try:
                ml_results = self.ner_pipeline(chunk_text)

                for ent in ml_results:
                    if ent["score"] >= self.min_score:
                        entities.append({
                            "start": ent["start"] + current_offset,
                            "end": ent["end"] + current_offset,
                            "entity_type": self._normalize_type(ent["entity_group"]),
                            "score": ent["score"],
                            "source": "ml"
                        })
            except Exception as e:
                print(f"⚠️ Erreur sur chunk à offset {current_offset}: {e}")

            current_offset += len(chunk_text)

        return entities

    def _split_text_smart(self, text: str, max_length: int) -> List[str]:
        """
        Découpe le texte intelligemment :
        - Priorité aux phrases complètes
        - Sinon sur des lignes (SPEAKER_X)
        - En dernier recours sur des mots
        """
        chunks = []

        # D'abord essaie de découper sur SPEAKER_X (typique ASR)
        speaker_pattern = r'(SPEAKER_\d+\s*:)'
        parts = re.split(speaker_pattern, text)

        current_chunk = ""
        for part in parts:
            if len(current_chunk) + len(part) <= max_length:
                current_chunk += part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        # Si un chunk est encore trop long, découpe sur des phrases
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Découpe sur les phrases
                sentences = re.split(r'([.!?]\s+)', chunk)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) <= max_length:
                        current += sent
                    else:
                        if current:
                            final_chunks.append(current)
                        current = sent
                if current:
                    final_chunks.append(current)

        return final_chunks

    def _find_domain_entities(self, text: str) -> List[Dict[str, Any]]:
        """Cherche les entités connues de ton domaine"""
        entities = []

        for entity_type, entity_list in self.domain_entities.items():
            for entity_value in entity_list:
                # Recherche avec word boundaries (\b) pour éviter faux positifs
                pattern = re.compile(r'\b' + re.escape(entity_value) + r'\b', re.IGNORECASE)

                for match in pattern.finditer(text):
                    entities.append({
                        "start": match.start(),
                        "end": match.end(),
                        "entity_type": entity_type,
                        "score": 1.0,
                        "source": "domain"
                    })

        return entities

    def _find_contextual_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Patterns contextuels pour transcriptions françaises"""
        entities = []

        # Personnes avec contexte
        person_patterns = [
            r"(?:avec|chez|par)\s+([A-Z][a-zàâäéèêëïîôùûü]+)",
            r"(?:M\.|Mme|Mlle|Dr|Me)\s+([A-Z][a-zàâäéèêëïîôùûü]+)",
        ]

        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    "start": match.start(1),
                    "end": match.end(1),
                    "entity_type": "PERSON",
                    "score": 0.75,
                    "source": "pattern"
                })

        # Organisations
        org_patterns = [
            r"\b(?:Cabinet|Société|SA|SARL|SAS)\s+([A-Z][a-zA-Zàâäéèêëïîôùûü\s]+?)(?=\s|,|\.|$)",
        ]

        for pattern in org_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if len(name) > 3:  # Évite captures courtes
                    entities.append({
                        "start": match.start(1),
                        "end": match.end(1),
                        "entity_type": "ORGANIZATION",
                        "score": 0.8,
                        "source": "pattern"
                    })

        # Personnes en début de phrase (ex: "Marine a appelé ...")
        verbs = (
            "a",
            "est",
            "était",
            "sera",
            "serait",
            "avait",
        )
        verbs_pattern = "|".join(sorted(set(verbs)))
        leading_name_pattern = re.compile(
            rf"(?:(?<=^)|(?<=\n)|(?<=[\.\!\?]\s))([A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ][a-zàâäçéèêëîïôöùûüÿ'\-]+)\s+(?:{verbs_pattern})\b",
            flags=re.MULTILINE
        )

        for match in leading_name_pattern.finditer(text):
            candidate = match.group(1)
            candidate_lower = candidate.lower()

            if candidate_lower in self.whitelist:
                continue
            if candidate_lower in self._domain_values:
                continue

            entities.append({
                "start": match.start(1),
                "end": match.end(1),
                "entity_type": "PERSON",
                "score": 0.65,
                "source": "pattern_leading"
            })

        return entities

    def _deduplicate_entities(
        self,
        entities: List[Dict[str, Any]],
        text: str
    ) -> List[Dict[str, Any]]:
        """Supprime les doublons et applique les filtres"""

        # Filtre whitelist et longueur
        candidates: List[Dict[str, Any]] = []
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            surface_raw = text[start:end]
            surface = surface_raw.strip()

            if not surface:
                continue

            if self._is_inside_speaker_label(text, start, end):
                continue

            context_segment = text[max(0, start - 2):min(len(text), end + 8)].lower()
            if "speaker" in context_segment:
                continue

            etype_norm = self._normalize_type(ent["entity_type"])

            if len(surface) < self.min_length:
                continue
            if surface.lower() in self.whitelist:
                continue
            if surface.upper().startswith("SPEAKER"):
                continue
            if "speaker" in surface.lower():
                continue
            if self._should_skip_surface(surface, etype_norm):
                continue

            ent_copy = ent.copy()

            domain_label = self._domain_lookup.get(surface.lower())
            if domain_label:
                ent_copy["entity_type"] = domain_label
                etype_norm = domain_label
            else:
                refined_label = self._refine_entity_label(surface, etype_norm)
                ent_copy["entity_type"] = refined_label
                etype_norm = refined_label

            ent_copy["_surface_lower"] = surface.lower()
            candidates.append(ent_copy)

        label_priority = {"PERSON": 4, "ORGANIZATION": 4, "LOCATION": 3, "MISC": 1}
        best_stats: Dict[str, Tuple[int, float]] = {}
        best_labels: Dict[str, str] = {}
        for ent_copy in candidates:
            surface_key = ent_copy["_surface_lower"]
            label = ent_copy["entity_type"]
            priority = label_priority[label]
            score = float(ent_copy.get("score", 1.0))
            current_priority, current_score = best_stats.get(surface_key, (0, -1.0))
            if priority > current_priority or (priority == current_priority and score > current_score):
                best_labels[surface_key] = label
                best_stats[surface_key] = (priority, score)

        filtered: List[Dict[str, Any]] = []
        for ent_copy in candidates:
            surface_key = ent_copy.pop("_surface_lower")
            ent_copy["entity_type"] = best_labels.get(surface_key, ent_copy["entity_type"])
            filtered.append(ent_copy)

        # Trie par position
        filtered.sort(key=lambda e: (e["start"], -e["score"]))

        # Supprime les chevauchements (garde le meilleur score)
        final = []
        for ent in filtered:
            overlap = False
            for existing in final:
                if not (ent["end"] <= existing["start"] or ent["start"] >= existing["end"]):
                    overlap = True
                    break

            if not overlap:
                final.append(ent)

        return final

    def anonymize_with_tags(
        self,
        text: str,
        entities: List[Dict[str, Any]] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Anonymise le texte et retourne le mapping pour désanonymisation.
        """

        if entities is None:
            entities = self.analyze(text)

        entities_sorted = sorted(entities, key=lambda e: e["start"])

        # Compteurs et mapping
        seen: Dict[Tuple[str, str], str] = {}
        counters = {"PERSON": 1, "ORGANIZATION": 1, "LOCATION": 1, "MISC": 1}

        mapping: Dict[str, Any] = {
            "entities": {},
            "reverse_map": {},
            "stats": {"total": 0, "by_type": {}}
        }

        spans: List[Tuple[int, int, str, str]] = []

        for ent in entities_sorted:
            start = ent["start"]
            end = ent["end"]
            etype = ent["entity_type"]
            raw_surface = text[start:end]

            # Préserve les espaces autour en ne remplaçant que la partie utile
            leading_ws = len(raw_surface) - len(raw_surface.lstrip())
            trailing_ws = len(raw_surface) - len(raw_surface.rstrip())

            trimmed_start = start + leading_ws
            trimmed_end = end - trailing_ws

            if trimmed_start >= trimmed_end:
                continue

            surface = text[trimmed_start:trimmed_end]

            # Normalise le type
            etype_norm = self._normalize_type(etype)

            # Crée ou récupère le tag
            key = (etype_norm, surface.lower())
            if key in seen:
                tag = seen[key]
            else:
                if etype_norm not in counters:
                    counters[etype_norm] = 1

                tag = f"<{etype_norm}_{counters[etype_norm]}>"
                counters[etype_norm] += 1
                seen[key] = tag

                mapping["entities"][tag] = {
                    "label": etype_norm,
                    "values": [surface],
                    "source": ent.get("source", "ml"),
                    "score": float(ent.get("score", 1.0))
                }
                mapping["reverse_map"][tag] = surface

            spans.append((trimmed_start, trimmed_end, tag, surface))
            mapping["stats"]["total"] += 1

        # Remplace de la fin vers le début SANS manger les espaces
        spans.sort(key=lambda x: x[0], reverse=True)
        anonymized = text

        for start, end, tag, surface in spans:
            anonymized = anonymized[:start] + tag + anonymized[end:]

        # Harmonise les labels de locuteurs au format "SPEAKER_X :" pour matcher les standards tests
        anonymized = re.sub(r"(SPEAKER_\d+)\s*:(\s*)", r"\1 :\2", anonymized)

        # Stats
        for tag_info in mapping["entities"].values():
            label = tag_info["label"]
            mapping["stats"]["by_type"][label] = mapping["stats"]["by_type"].get(label, 0) + 1

        return anonymized, mapping

    def deanonymize(self, anonymized_text: str, mapping: Dict[str, Any]) -> str:
        """Désanonymise le texte de façon exacte"""
        result = anonymized_text

        for tag, original_value in mapping.get("reverse_map", {}).items():
            result = result.replace(tag, original_value)

        return result

    @staticmethod
    def _normalize_type(etype: str) -> str:
        """Normalise les types d'entités"""
        et = etype.upper()

        if et in ("PER", "PERSON"):
            return "PERSON"
        if et in ("ORG", "ORGANIZATION", "COMPANY"):
            return "ORGANIZATION"
        if et in ("LOC", "LOCATION", "GPE"):
            return "LOCATION"

        return "MISC"

    @staticmethod
    def _is_inside_speaker_label(text: str, start: int, end: int) -> bool:
        """Retourne True si le span appartient au préfixe SPEAKER_X: qu'on doit préserver."""
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", start)
        if line_end == -1:
            line_end = len(text)

        line_segment = text[line_start:line_end]
        match = re.match(r"\s*SPEAKER_\d+\s*:", line_segment)
        if not match:
            return False

        label_end = line_start + match.end()
        return start < label_end

    @staticmethod
    def _build_domain_lookup(domain_entities: Dict[str, List[str]]) -> Dict[str, str]:
        """Construit un mapping lowercase -> label pour les entités du domaine."""
        lookup: Dict[str, str] = {}
        for label, values in domain_entities.items():
            normalized_label = TransformerAnonymizer._normalize_type(label)
            for value in values:
                lookup[value.lower()] = normalized_label
        return lookup

    def _refine_entity_label(self, surface: str, label: str) -> str:
        """
        Ajuste le label d'une entité selon des heuristiques simples pour réduire les faux positifs.
        """
        clean = surface.strip()
        if not clean:
            return label

        label = self._normalize_type(label)
        lowercase = clean.lower()

        # Promote MISC to PERSON when it looks like a proper name (1-2 Title Case tokens)
        if label == "MISC":
            tokens = [token for token in re.split(r"[\\s\\-]+", clean) if token]

            def is_title(token: str) -> bool:
                return token[:1].isalpha() and token[0].isupper() and token[1:] == token[1:].lower()

            if tokens and len(tokens) <= 3 and all(is_title(tok) for tok in tokens):
                if lowercase not in self.whitelist:
                    return "PERSON"

        # Degrade PERSON/ORGANIZATION predictions when the surface is entirely lowercase (likely noise)
        if lowercase == clean:
            if label in {"PERSON", "ORGANIZATION"}:
                return "MISC"

        # Downgrade ORGANIZATION predictions for very short fragments
        if label == "ORGANIZATION" and len(clean) <= 3 and lowercase == clean:
            return "MISC"

        return label

    def _should_skip_surface(self, surface: str, etype_norm: str) -> bool:
        """Heuristiques simples pour ignorer les faux positifs évidents."""
        clean = surface.strip()
        if not clean:
            return True

        clean_lower = clean.lower()

        # Ignore duplicates for domain values handled elsewhere
        if clean_lower in self._domain_values:
            return False

        # Évite les fragments de phrases génériques
        tokens = clean.split()
        uppercase_tokens = sum(1 for tok in tokens if tok and tok[0].isupper())

        if len(tokens) > 3 and uppercase_tokens == 0:
            return True

        # Pour les spans multi-mots sans majuscule, ignore (souvent du texte générique)
        if len(tokens) > 1 and uppercase_tokens == 0:
            return True

        # Pour MISC, évite les mots entièrement en minuscules (sms, mail, etc.)
        if etype_norm == "MISC" and clean.islower():
            return True

        # Ignore les mots très courts entièrement en minuscules
        if len(clean) <= 2 and clean_lower not in self._domain_values:
            return True

        if clean.islower() and len(tokens) == 1 and len(clean) <= 3:
            return True

        # Ignore les chaînes qui contiennent peu de lettres (souvent bruit)
        letters = sum(1 for ch in clean if ch.isalpha())
        if letters <= 1:
            return True

        return False


def run_transformer_anonymization(
    text: str,
    domain_entities: Dict[str, List[str]] | None = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Point d'entrée simple, compatible avec run_presidio_anonymization()
    """
    anonymizer = TransformerAnonymizer(domain_entities=domain_entities)
    return anonymizer.anonymize_with_tags(text)
