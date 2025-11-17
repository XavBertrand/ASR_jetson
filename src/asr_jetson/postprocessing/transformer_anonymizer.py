# src/asr_jetson/postprocessing/transformer_anonymizer.py

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz
from unidecode import unidecode
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
        self._domain_canonical = {
            value.lower(): value for values in self.domain_entities.values() for value in values
        }

        # Seuils
        self.min_score = 0.5
        self.min_length = 2
        # Stopwords et mots génériques à ignorer pour éviter les faux positifs.
        self.person_blocklist = {
            "elle", "elles", "il", "ils", "on", "je", "tu", "nous", "vous", "toi", "moi",
            "lui", "leur", "leurs", "quelqu'un", "personne", "personnes", "chacun", "aucun",
            "quand", "pourquoi", "comment", "merci", "bonjour", "soir", "matin", "midi",
            "relance", "relances", "relancer", "moral", "morale", "cooptation", "rupture", "serfa",
            "serfas", "cerfa", "cerfas"
        }
        self.generic_blocklist = self.person_blocklist | {
            "sms", "whatsapp", "mail", "mails", "email", "emails", "tennis", "lundi", "mardi",
            "mercredi", "jeudi", "vendredi", "samedi", "dimanche"
        }

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
                    surface = text[match.start():match.end()]
                    if self._looks_like_common_noun_usage(surface, text, match.start(), match.end()):
                        continue

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
            surface = self._clean_surface(surface_raw)

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
            surface_lower = surface.lower()
            if surface_lower in self.whitelist:
                continue
            if surface.upper().startswith("SPEAKER"):
                continue
            if "speaker" in surface.lower():
                continue
            if self._should_skip_surface(surface, etype_norm):
                continue
            common_usage = self._looks_like_common_noun_usage(surface, text, start, end)

            ent_copy = ent.copy()
            tokens_lower = {tok.lower() for tok in re.split(r"[\s,\.;:!\?\-\(\)]+", surface) if tok}
            if etype_norm == "PERSON" and (
                surface_lower in self.person_blocklist or tokens_lower & self.person_blocklist
            ):
                continue
            if tokens_lower & self.generic_blocklist:
                continue
            domain_label = self._domain_lookup.get(surface.lower())
            if domain_label and not common_usage:
                ent_copy["entity_type"] = domain_label
                etype_norm = domain_label
            else:
                refined_label = self._refine_entity_label(surface, etype_norm)
                ent_copy["entity_type"] = refined_label
                etype_norm = refined_label

            ent_copy["_surface_lower"] = surface_lower
            ent_copy["_surface_is_lower"] = surface.islower()
            ent_copy["_looks_common"] = common_usage
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
            surface_is_lower = ent_copy.pop("_surface_is_lower", False)
            looks_common = ent_copy.pop("_looks_common", False)
            original_label = ent_copy["entity_type"]
            resolved_label = best_labels.get(surface_key, original_label)

            if (
                resolved_label == "PERSON"
                and original_label != "PERSON"
                and surface_is_lower
                and looks_common
            ):
                # Conserve la mention générique (ex: "marine" en tant que mer)
                # lorsque seule la casse différencie les usages.
                continue

            ent_copy["entity_type"] = resolved_label
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
        else:
            entities = self._deduplicate_entities(entities, text)

        entities_sorted = sorted(entities, key=lambda e: e["start"])

        counters = {"PERSON": 1, "ORGANIZATION": 1, "LOCATION": 1, "MISC": 1}
        mapping: Dict[str, Any] = {
            "entities": {},
            "reverse_map": {},
            "stats": {"total": 0, "by_type": {}},
        }
        spans: List[Tuple[int, int, str]] = []
        grouped_tags: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for ent in entities_sorted:
            start = ent["start"]
            end = ent["end"]
            etype = self._normalize_type(ent["entity_type"])
            raw_surface = text[start:end]

            leading_ws = len(raw_surface) - len(raw_surface.lstrip())
            trailing_ws = len(raw_surface) - len(raw_surface.rstrip())
            trimmed_start = start + leading_ws
            trimmed_end = end - trailing_ws
            if trimmed_start >= trimmed_end:
                continue

            surface = text[trimmed_start:trimmed_end]
            normalized_display = self._prepare_surface_for_mapping(surface, etype)
            norm_compact, norm_spaced = self._normalize_surface_key(normalized_display)
            if not norm_compact and not norm_spaced:
                continue

            existing = self._find_existing_group(grouped_tags[etype], norm_compact, norm_spaced)
            if existing:
                tag = existing["tag"]
                info = mapping["entities"][tag]
                best_value = self._pick_better_surface(
                    info.get("canonical", info["values"][0]),
                    normalized_display,
                    etype,
                    current_score=info.get("score", 1.0),
                    candidate_score=float(ent.get("score", 1.0)),
                )
                if best_value != info.get("canonical"):
                    info["canonical"] = best_value
                    mapping["reverse_map"][tag] = best_value
                    existing["canonical"] = best_value
                    compact_new, spaced_new = self._normalize_surface_key(best_value)
                    existing["normalized_compact"] = compact_new
                    existing["normalized_spaced"] = spaced_new
                if normalized_display not in info["values"]:
                    info["values"].append(normalized_display)
                variants = info.setdefault("variants", [])
                if surface not in variants:
                    variants.append(surface)
                new_score = float(ent.get("score", 1.0))
                if new_score > info.get("score", 0.0):
                    info["score"] = new_score
                    info["source"] = ent.get("source", info.get("source", "ml"))
            else:
                if etype not in counters:
                    counters[etype] = 1
                tag = f"<{etype}_{counters[etype]}>"
                counters[etype] += 1
                canonical_value = normalized_display
                mapping["entities"][tag] = {
                    "label": etype,
                    "values": [canonical_value],
                    "variants": [surface],
                    "source": ent.get("source", "ml"),
                    "score": float(ent.get("score", 1.0)),
                    "canonical": canonical_value,
                }
                mapping["reverse_map"][tag] = canonical_value
                grouped_tags[etype].append(
                    {
                        "tag": tag,
                        "normalized_compact": norm_compact,
                        "normalized_spaced": norm_spaced,
                        "canonical": canonical_value,
                    }
                )

            spans.append((trimmed_start, trimmed_end, tag))
            mapping["stats"]["total"] += 1

        replacements_corrected = [
            (start, end, mapping["reverse_map"][tag]) for start, end, tag in spans
        ]
        mapping["corrected_text"] = self._apply_replacements(text, replacements_corrected)

        replacements_tags = [(start, end, tag) for start, end, tag in spans]
        anonymized = self._apply_replacements(text, replacements_tags)

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

    def _clean_surface(self, surface: str) -> str:
        """Supprime les guillemets et espaces superflus autour d'un span."""
        clean = surface.strip()
        clean = re.sub(r"^[\"'“”‘’\(\)\[\]]+", "", clean)
        clean = re.sub(r"[\"'“”‘’\)\(\[\]]+$", "", clean)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _prepare_surface_for_mapping(self, surface: str, label: str) -> str:
        """Normalise le rendu (casse, accents) pour homogénéiser le mapping."""
        clean = self._clean_surface(surface)
        if not clean:
            return ""

        domain_value = self._domain_canonical.get(clean.lower())
        if domain_value:
            return domain_value

        if label == "PERSON":
            tokens = [tok for tok in re.split(r"\s+", clean) if tok]
            normalized_tokens = []
            for tok in tokens:
                ascii_token = unidecode(tok)
                if ascii_token.isupper() and len(ascii_token) <= 4:
                    normalized_tokens.append(ascii_token)
                elif len(ascii_token) > 1:
                    normalized_tokens.append(ascii_token[0].upper() + ascii_token[1:].lower())
                else:
                    normalized_tokens.append(ascii_token.upper())
            return " ".join(normalized_tokens)

        if label in {"ORGANIZATION", "LOCATION"} and clean.islower():
            return clean.title()

        return clean

    def _find_existing_group(
        self,
        groups: List[Dict[str, Any]],
        norm_compact: str,
        norm_spaced: str,
    ) -> Optional[Dict[str, Any]]:
        """Retourne le groupe existant dont la clé normalisée est proche."""
        for group in groups:
            if self._surfaces_close(
                group.get("normalized_compact", ""),
                norm_compact,
                group.get("normalized_spaced", ""),
                norm_spaced,
            ):
                return group
        return None

    def _pick_better_surface(
        self,
        current: str,
        candidate: str,
        label: str,
        *,
        current_score: float = 1.0,
        candidate_score: float = 1.0,
    ) -> str:
        """Choisit la meilleure variante à conserver comme canonique."""
        candidate = self._prepare_surface_for_mapping(candidate, label)
        if not current:
            return candidate
        if not candidate:
            return current
        if candidate.lower() in self._domain_canonical:
            return self._domain_canonical[candidate.lower()]
        if current.lower() in self._domain_canonical:
            return current

        curr_compact, _ = self._normalize_surface_key(current)
        cand_compact, _ = self._normalize_surface_key(candidate)
        if label == "PERSON":
            if curr_compact.endswith("s") and curr_compact.rstrip("s") == cand_compact:
                return candidate
            if cand_compact.endswith("s") and cand_compact.rstrip("s") == curr_compact:
                return current

        if candidate_score > current_score + 0.05:
            return candidate
        if current_score > candidate_score + 0.05:
            return current

        curr_quality = self._surface_quality_score(current)
        cand_quality = self._surface_quality_score(candidate)
        return candidate if cand_quality > curr_quality else current

    @staticmethod
    def _tokenize_window(window: str) -> List[str]:
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", window)

    def _looks_like_common_noun_usage(self, surface: str, text: str, start: int, end: int) -> bool:
        """
        Détecte les contextes typiques d'un nom commun (ex: 'la vie marine').
        Utilisé pour éviter de transformer des homophones en PERSON.
        """
        clean = surface.strip()
        if not clean or not clean.islower():
            return False

        window_before = text[max(0, start - 30):start]
        window_after = text[end:min(len(text), end + 30)]
        before_tokens = self._tokenize_window(window_before)
        after_tokens = self._tokenize_window(window_after)

        prev_token = unidecode(before_tokens[-1]).lower() if before_tokens else ""
        prev_prev = unidecode(before_tokens[-2]).lower() if len(before_tokens) >= 2 else ""
        next_token = unidecode(after_tokens[0]).lower() if after_tokens else ""

        determiners = {"la", "le", "les", "une", "un", "des", "du", "de", "d", "l", "au", "aux"}
        noun_contexts = {
            "vie", "faune", "flore", "ressource", "ressources", "biodiversite", "biodiversité",
            "milieu", "milieux", "industrie", "industries", "ministere", "ministeres",
            "arme", "armee", "marine", "justice", "police"
        }
        following_contexts = {
            "fragile", "fragiles", "durable", "durables", "maritime", "maritimes",
            "marchande", "marchandes", "nationale", "nationales", "militaire", "militaires",
            "protegee", "protegees", "protégée", "protégées", "sensible", "sensibles"
        }

        if prev_token in determiners:
            return True
        if prev_token in noun_contexts or prev_prev in noun_contexts:
            return True
        if prev_token in {"de", "d"} and prev_prev in determiners:
            return True
        if next_token in following_contexts:
            return True

        return False

    @staticmethod
    def _surface_quality_score(surface: str) -> Tuple[int, int, int, int]:
        """Heuristique simple pour prioriser les variantes (casse, longueur)."""
        clean = surface.strip()
        tokens = [tok for tok in re.split(r"\s+", clean) if tok]
        proper_tokens = sum(1 for tok in tokens if tok and tok[0].isupper())
        has_upper = int(any(ch.isupper() for ch in clean))
        has_lower = int(any(ch.islower() for ch in clean))
        return (
            proper_tokens,
            len(tokens),
            has_upper - (0 if has_lower else 1),
            len(clean),
        )

    @staticmethod
    def _normalize_surface_key(surface: str) -> Tuple[str, str]:
        base = unidecode(surface.lower())
        compact = re.sub(r"[^a-z0-9]", "", base)
        spaced = re.sub(r"[^a-z0-9]+", " ", base).strip()
        return compact, spaced

    @staticmethod
    def _surfaces_close(
        norm_a_compact: str,
        norm_b_compact: str,
        norm_a_spaced: str,
        norm_b_spaced: str,
    ) -> bool:
        if not norm_a_compact or not norm_b_compact:
            return False
        if norm_a_compact == norm_b_compact:
            return True

        ratios = [
            fuzz.ratio(norm_a_compact, norm_b_compact),
            fuzz.partial_ratio(norm_a_compact, norm_b_compact),
        ]
        spaced_scores = [
            fuzz.token_set_ratio(norm_a_spaced, norm_b_spaced) if norm_a_spaced and norm_b_spaced else 0,
            fuzz.token_sort_ratio(norm_a_spaced, norm_b_spaced) if norm_a_spaced and norm_b_spaced else 0,
        ]
        if max(ratios + spaced_scores) >= 92:
            return True
        if (
            norm_a_compact[0] == norm_b_compact[0]
            and abs(len(norm_a_compact) - len(norm_b_compact)) <= 3
            and max(ratios + spaced_scores) >= 80
        ):
            return True
        return False

    @staticmethod
    def _apply_replacements(text: str, replacements: List[Tuple[int, int, str]]) -> str:
        """Applique une liste de remplacements sans décaler les index suivants."""
        if not replacements:
            return text
        result = text
        for start, end, value in sorted(replacements, key=lambda x: x[0], reverse=True):
            result = result[:start] + value + result[end:]
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
        if lowercase == clean and label == "ORGANIZATION":
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
        if clean_lower in self.generic_blocklist:
            return True

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
