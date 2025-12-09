from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import asr_jetson.postprocessing.mistral_client as mistral_client
from asr_jetson.postprocessing.anonymizer import deanonymize_text
import requests  # type: ignore

try:  # pragma: no cover - optional dependency
    import pypandoc  # type: ignore

    _HAS_PYPANDOC = True
    _PYPANDOC_IMPORT_ERROR: Optional[Exception] = None
except Exception as _err:  # pragma: no cover - executed when pypandoc missing
    pypandoc = None  # type: ignore
    _HAS_PYPANDOC = False
    _PYPANDOC_IMPORT_ERROR = _err

try:  # pragma: no cover - optional dependency
    from weasyprint import CSS, HTML  # type: ignore

    _HAS_WEASYPRINT = True
    _WEASYPRINT_IMPORT_ERROR: Optional[Exception] = None
except Exception as _err:  # pragma: no cover - executed when weasyprint missing
    CSS = None  # type: ignore
    HTML = None  # type: ignore
    _HAS_WEASYPRINT = False
    _WEASYPRINT_IMPORT_ERROR = _err

_TAG_NORM_RE = re.compile(r"<\s*([a-zA-Z]+)\s*_(\s*\d+)\s*[^>]*>|\{\s*([a-zA-Z]+)\s*_(\s*\d+)\s*[^}]*\}", re.UNICODE)
_PERSON_TAG_RE = re.compile(r"<\s*PERSON\s*_(\s*\d+)\s*>", re.IGNORECASE)
_PANDOC_MD_FORMAT = (
    "markdown+pipe_tables+grid_tables+multiline_tables+table_captions+raw_html+fenced_divs"
    "-yaml_metadata_block"
)
_PREFERRED_SANS_FONTS: Tuple[str, ...] = ("Arial", "Calibri", "Liberation Sans", "DejaVu Sans")
_REPORT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "config" / "meeting.html"
_REPORT_CSS_PATH = Path(__file__).resolve().parent.parent / "config" / "report.css"
_ROLE_KEYWORD_HINTS = {
    "delphine": "Avocat gérante",
    "marie": "Avocat gérante",
    "marine": "Collaborateur",
    "sylvie": "Collaborateur",
}
_PERSON_CANONICAL_ALIASES = {
    "delphine": "Delphine",
    "marie": "Delphine",
    "marine": "Marine",
    "sylvie": "Sylvie",
    "marine.": "Marine",
}
_DEFAULT_MISTRAL_HEALTHCHECK_URL = "https://api.mistral.ai/v1/models"
_MAIN_SECTIONS = [
    "RÉSUMÉ EXÉCUTIF",
    "PARTICIPANTS",
    "SUJETS ABORDÉS",
    "DÉCISIONS",
    "ACTIONS",
    "PROCHAINES ÉTAPES",
    "ANALYSE DES RISQUES ET OPPORTUNITÉS",
    "CHIFFRES ET REPÈRES TEMPORELS",
    "POINTS POSITIFS EXPRIMÉS",
    "POINTS DE FRICTION OU DIFFICULTÉS",
]
_MAIN_SECTIONS_REGEX = "|".join(re.escape(title) for title in _MAIN_SECTIONS)


def _check_mistral_access(timeout: float = 5.0) -> Tuple[bool, str]:
    """
    Vérifie que l'API Mistral est accessible avec la configuration courante.
    Retourne (True, "") si OK, sinon (False, raison).
    """
    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
    if not api_key:
        return False, "MISTRAL_API_KEY absent de l'environnement."
    if mistral_client.Mistral is None:
        return False, "Le package 'mistralai' est requis pour contacter l'API Mistral."

    healthcheck_url = os.getenv("MISTRAL_HEALTHCHECK_URL", _DEFAULT_MISTRAL_HEALTHCHECK_URL)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    try:
        response = requests.get(healthcheck_url, headers=headers, timeout=timeout)
    except Exception as exc:  # pragma: no cover - dépend du réseau
        return False, f"API Mistral inaccessible ({exc})"

    if response.status_code >= 400:
        return False, f"API Mistral non disponible (status={response.status_code})"
    return True, ""


def _empty_report_outputs(reason: str, status: str = "skipped") -> Dict[str, Optional[str]]:
    return {
        "report_anonymized_txt": None,
        "report_txt": None,
        "report_markdown": None,
        "report_docx": None,
        "report_pdf": None,
        "report_status": status,
        "report_reason": reason,
    }


def _collect_pseudonyms_by_label(mapping: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Regroupe les pseudonymes par label (PERSON, ORGANIZATION, etc.) sans exposer
    les noms réels. S'appuie sur la structure produite par TransformerAnonymizer.
    """

    def _add(label: str, pseudonym: Optional[str]) -> None:
        if not pseudonym:
            return
        cleaned = pseudonym.strip()
        if not cleaned:
            return
        normalized_label = (label or "ENTITÉ").strip() or "ENTITÉ"
        bucket = collected.setdefault(normalized_label.upper(), [])
        if cleaned not in bucket:
            bucket.append(cleaned)

    collected: Dict[str, List[str]] = {}
    entities = mapping.get("entities")

    if isinstance(entities, dict):
        for info in entities.values():
            _add(info.get("label") or info.get("type"), info.get("pseudonym"))
    elif isinstance(entities, list):
        for info in entities:
            _add(info.get("label") or info.get("type"), info.get("pseudonym"))

    if not collected:
        pseudo_map = mapping.get("pseudonym_map", {})
        if isinstance(pseudo_map, dict):
            for pseudonym in pseudo_map.values():
                _add("ENTITÉ", pseudonym)

    return collected


def _build_pseudonym_hint(mapping: Dict[str, Any]) -> str:
    """
    Construit un rappel textuel des pseudonymes à utiliser pour forcer le LLM
    à rester dans l'espace anonymisé.
    """
    grouped = _collect_pseudonyms_by_label(mapping)
    if not grouped:
        return ""

    lines = [
        "Consigne anonymisation : tous les noms ci-dessous sont des pseudonymes.",
        "Recopie-les tels quels et ne tente jamais de deviner les noms réels.",
        "",
        "Pseudonymes détectés :",
    ]

    for label in sorted(grouped.keys()):
        values = grouped[label]
        if not values:
            continue
        display_label = label.title()
        joined = ", ".join(values)
        lines.append(f"- {display_label} : {joined}")

    lines.append("")
    return "\n".join(lines)


def _build_speakers_hint(speaker_labels: Set[str]) -> str:
    """
    Liste les locuteurs réellement détectés (SPEAKER_1, SPK2, etc.) pour forcer
    le LLM à ne lister que ces personnes dans la section PARTICIPANTS.
    """
    if not speaker_labels:
        return ""
    ordered = sorted(speaker_labels)
    listed = ", ".join(ordered)
    return (
        "Locuteurs détectés dans la transcription (uniquement ceux-ci doivent apparaître"
        f" dans PARTICIPANTS) : {listed}.\nNe pas ajouter de personnes simplement citées.\n\n"
    )


def _format_speaker_context(context: Optional[str]) -> str:
    """
    Mise en forme du contexte locuteurs (déjà anonymisé) pour le prompt LLM.
    """
    if not context:
        return ""
    cleaned = context.strip()
    if not cleaned:
        return ""
    return (
        "Contexte sur les interlocuteurs (pseudonymisé, fourni par l'utilisateur) :\n"
        f"{cleaned}\n\n"
    )


def _normalize_title_key(text: str) -> str:
    """
    Supprime les accents et harmonise la casse pour comparer les titres de section.
    """
    normalized = unicodedata.normalize("NFD", text or "")
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return normalized.upper().strip()


def _split_section_header_line(line: str) -> Tuple[str, str]:
    """
    Sépare un éventuel contenu placé sur la même ligne que le titre de section.
    Retourne (header, trailing_content).
    """
    stripped = line.strip()
    if not stripped.startswith("###"):
        return stripped, ""

    # ### 1. RÉSUMÉ EXÉCUTIF du texte supplémentaire
    match = re.match(
        rf"^(###\s+(?:\d+[.)]\s+)?)(?:({_MAIN_SECTIONS_REGEX}))(.*)$",
        stripped,
        flags=re.IGNORECASE,
    )
    if not match:
        return stripped, ""

    prefix = match.group(1).strip()
    raw_title = match.group(2).strip()
    trailing = (match.group(3) or "").strip()

    normalized_title = _normalize_title_key(raw_title)
    canonical_title = next(
        (title for title in _MAIN_SECTIONS if _normalize_title_key(title) == normalized_title),
        raw_title,
    )
    header = f"{prefix} {canonical_title}".strip()
    return header, trailing


def _normalize_llm_placeholders(text: str) -> str:
    """
    - Uniformise <org_9>, <Org_9>, <ORG_9>, et même <ORG_9...> en <ORG_9>
    - Corrige les accolade/chevrons mal fermés (<PER_1}) → <PER_1>
    - Supprime espaces parasites dans l’index (<ORG_ 9> → <ORG_9>)
    """
    def _repl(m):
        # match peut être avec chevrons (groups 1,2) ou accolades (groups 3,4)
        if m.group(1) and m.group(2):
            typ = m.group(1).upper()
            idx = re.sub(r"\s+", "", m.group(2))
            return f"<{typ}_{idx}>"
        else:
            typ = m.group(3).upper()
            idx = re.sub(r"\s+", "", m.group(4))
            return f"<{typ}_{idx}>"
    # remplace aussi chevrons/ou accolades de fin foireux
    text = text.replace("}>", ">").replace("}", ">")
    return _TAG_NORM_RE.sub(_repl, text)


def _ensure_legacy_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convertit la structure de mapping produite par ``TransformerAnonymizer``
    (dictionnaire par tag) en format historique attendu par ``deanonymize_text``.
    """
    entities = mapping.get("entities")
    if not isinstance(entities, dict):
        return mapping

    reverse_map = mapping.get("reverse_map", {})
    new_entities: List[Dict[str, Any]] = []

    for tag, info in entities.items():
        values = list(dict.fromkeys(info.get("values", [])))
        canonical = reverse_map.get(tag) or (values[0] if values else None)
        entity_data: Dict[str, Any] = {
            "tag": tag,
            "type": info.get("label"),
            "canonical": canonical,
            "mentions": values,
            "sources": [info["source"]] if info.get("source") else [],
        }
        if "score" in info and info["score"] is not None:
            try:
                entity_data["score_avg"] = round(float(info["score"]), 4)
            except (TypeError, ValueError):
                pass
        new_entities.append(entity_data)

    converted = mapping.copy()
    converted["entities"] = new_entities
    if reverse_map:
        converted.setdefault("tag_lookup", dict(reverse_map))
    return converted


def _normalize_tag_key(tag: str) -> str:
    raw = (tag or "").strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1]
    return raw.strip().upper()


def _normalize_tag(tag: str) -> str:
    normalized = _normalize_tag_key(tag)
    if not normalized:
        return ""
    return f"<{normalized}>"


def _build_tag_lookup(mapping: Dict[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for key in ("tag_lookup", "reverse_map"):
        source = mapping.get(key)
        if isinstance(source, dict):
            for tag, canonical in source.items():
                if canonical:
                    lookup[_normalize_tag_key(tag)] = canonical
    entities = mapping.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            tag = entity.get("tag")
            canonical = entity.get("canonical")
            if tag and canonical and _normalize_tag_key(tag) not in lookup:
                lookup[_normalize_tag_key(tag)] = canonical
    return lookup


def _collect_known_tags(mapping: Dict[str, Any]) -> Set[str]:
    tags: Set[str] = set()
    for key in ("tag_lookup", "reverse_map"):
        source = mapping.get(key)
        if isinstance(source, dict):
            tags.update(_normalize_tag_key(tag) for tag in source.keys())
    entities = mapping.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            tag = entity.get("tag")
            if tag:
                tags.add(_normalize_tag_key(tag))
    return tags


def _log_name_resolution(mapping: Dict[str, Any]) -> None:
    """
    Trace dans la console les correspondances pseudonyme -> nom canoniques
    (limité aux PERSON pour rester lisible).
    """
    pseudo_reverse = mapping.get("pseudonym_reverse_map", {})
    if not isinstance(pseudo_reverse, dict) or not pseudo_reverse:
        return

    pseudo_labels: Dict[str, str] = {}
    entities = mapping.get("entities")
    if isinstance(entities, dict):
        for info in entities.values():
            label = (info.get("label") or info.get("type") or "").upper()
            pseudo = info.get("pseudonym")
            if pseudo and label:
                pseudo_labels[pseudo] = label
    elif isinstance(entities, list):
        for info in entities:
            label = (info.get("label") or info.get("type") or "").upper()
            pseudo = info.get("pseudonym")
            if pseudo and label:
                pseudo_labels[pseudo] = label

    lines: List[str] = []
    for pseudo, canonical in sorted(pseudo_reverse.items()):
        if not canonical or pseudo == canonical:
            continue
        label = pseudo_labels.get(pseudo, "")
        if label and label != "PERSON":
            continue
        lines.append(f"{pseudo} -> {canonical}")

    if lines:
        print("[rapport] Résolution des locuteurs :")
        for line in lines:
            print(f"  {line}")


def _count_person_tags(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for match in _PERSON_TAG_RE.finditer(text):
        key = _normalize_tag_key(match.group(0))
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _extract_speaker_labels(text: str) -> Set[str]:
    labels: Set[str] = set()
    for pattern in (
        r"\bSPEAKER[_\-\s]?(\d+)\b",
        r"\bSPK[_\-\s]?(\d+)\b",
        r"\bLOCUTEUR[_\-\s]?(\d+)\b",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            idx = match.group(1).lstrip("0") or match.group(1)
            labels.add(f"speaker_{idx.lower()}")
    return labels


def _canonical_entities_by_label(mapping: Dict[str, Any]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    entities = mapping.get("entities")
    if not isinstance(entities, list):
        return grouped
    for entity in entities:
        canonical = (entity.get("canonical") or "").strip()
        if not canonical:
            continue
        label = (entity.get("type") or entity.get("label") or "ENTITÉ").strip()
        if not label:
            label = "ENTITÉ"
        normalized_label = label.upper()
        grouped.setdefault(normalized_label, [])
        grouped[normalized_label].append(canonical)
    for label, names in grouped.items():
        deduped: List[str] = []
        seen: Set[str] = set()
        for name in names:
            lowered = name.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(name)
        grouped[label] = deduped
    return grouped


def _select_relevant_person_tags(
    mapping: Dict[str, Any],
    counts: Dict[str, int],
    max_tags: int = 6,
) -> Set[str]:
    relevant: Set[str] = set()
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    for tag_key, count in sorted_counts:
        if count >= 2:
            relevant.add(tag_key)
        if len(relevant) >= max_tags:
            break

    reverse_map = mapping.get("reverse_map", {})
    if isinstance(reverse_map, dict):
        for raw_tag, canonical in reverse_map.items():
            normalized_tag = _normalize_tag_key(raw_tag)
            canonical_lower = (canonical or "").strip().lower()
            if not normalized_tag or not canonical_lower:
                continue
            for keyword, role in _ROLE_KEYWORD_HINTS.items():
                if keyword in canonical_lower:
                    relevant.add(normalized_tag)
    if not relevant:
        relevant.update(tag for tag, _ in sorted_counts[:max_tags])
    return relevant


def _select_speaker_tags(
    mapping: Dict[str, Any],
    counts: Dict[str, int],
    max_speakers: Optional[int] = None,
) -> Set[str]:
    """
    Sélectionne un sous-ensemble de tags PERSON supposés correspondre aux locuteurs
    réellement présents. Utilise la fréquence des tags dans la transcription puis
    un éventuel fallback sur le mapping.
    """
    if max_speakers is not None and max_speakers <= 0:
        return set()

    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    selected: List[str] = []
    max_allowed = max_speakers or 3

    for tag_key, _ in sorted_counts:
        normalized = _normalize_tag_key(tag_key)
        if not normalized:
            continue
        if normalized not in selected:
            selected.append(normalized)
        if len(selected) >= max_allowed:
            break

    if len(selected) < max_allowed:
        reverse_map = mapping.get("reverse_map", {})
        if isinstance(reverse_map, dict):
            for raw_tag in reverse_map.keys():
                normalized = _normalize_tag_key(raw_tag)
                if normalized and normalized not in selected:
                    selected.append(normalized)
                if len(selected) >= max_allowed:
                    break

    return set(selected)


def _rationalize_person_tags(text: str, mapping: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    reverse_map = mapping.get("reverse_map")
    entities = mapping.get("entities")
    if not isinstance(reverse_map, dict) or not isinstance(entities, dict):
        return text, mapping

    canonical_to_tag: Dict[str, str] = {}
    replacements: Dict[str, str] = {}

    for raw_tag, canonical in reverse_map.items():
        if not isinstance(raw_tag, str) or "PERSON" not in raw_tag.upper():
            continue
        normalized_tag = _normalize_tag(raw_tag)
        if not normalized_tag:
            continue
        canonical_clean = (canonical or "").strip()
        canonical_key = canonical_clean.lower() if canonical_clean else normalized_tag
        primary_tag = canonical_to_tag.setdefault(canonical_key, normalized_tag)
        if primary_tag != normalized_tag:
            replacements[normalized_tag] = primary_tag

    if replacements:
        for src_tag, dst_tag in replacements.items():
            if src_tag == dst_tag:
                continue
            text = text.replace(src_tag, dst_tag)

    new_reverse_map: Dict[str, str] = {}
    normalized_alias_lookup: Dict[str, str] = {}
    for raw_tag, canonical in reverse_map.items():
        if not isinstance(raw_tag, str):
            continue
        normalized_tag = _normalize_tag(raw_tag)
        if not normalized_tag:
            continue
        normalized_tag = replacements.get(normalized_tag, normalized_tag)
        canonical_clean = (canonical or "").strip()
        if canonical_clean and normalized_tag not in new_reverse_map:
            new_reverse_map[normalized_tag] = canonical_clean
        alias_display = _PERSON_CANONICAL_ALIASES.get(canonical_clean.lower())
        if (
            alias_display
            and alias_display.strip()
            and alias_display.strip().lower() != canonical_clean.lower()
        ):
            normalized_alias_lookup[normalized_tag] = alias_display.strip()

    new_entities: Dict[str, Any] = {}
    for raw_tag, info in entities.items():
        normalized_tag = _normalize_tag(raw_tag)
        if not normalized_tag:
            continue
        normalized_tag = replacements.get(normalized_tag, normalized_tag)
        existing = new_entities.get(normalized_tag)
        info_copy = dict(info)
        reverse_value = new_reverse_map.get(normalized_tag)
        if reverse_value and isinstance(info_copy.get("values"), list):
            values = [reverse_value] + [val for val in info_copy["values"] if val != reverse_value]
            info_copy["values"] = list(dict.fromkeys(values))
        if existing:
            merged_values = list(
                dict.fromkeys(
                    (existing.get("values", []) or [])
                    + (info_copy.get("values", []) or [])
                )
            )
            existing["values"] = merged_values
        else:
            new_entities[normalized_tag] = info_copy

    mapping["reverse_map"] = new_reverse_map
    mapping["entities"] = new_entities
    if normalized_alias_lookup:
        mapping["alias_lookup"] = normalized_alias_lookup
    return text, mapping


def _normalize_role_label(raw: str) -> str:
    lowered = raw.lower()
    if "avocat" in lowered and ("gerant" in lowered or "gérant" in lowered or "gérante" in lowered):
        return "Avocat gérante"
    if "delphine" in lowered:
        return "Avocat gérante"
    if "collaborateur" in lowered or "collaboratrice" in lowered:
        return "Collaborateur"
    if "assistant" in lowered or "assistante" in lowered or "secrétaire" in lowered:
        return "Collaborateur"
    if "client" in lowered or "cliente" in lowered:
        return "Client"
    return raw.strip()


def _extract_role_hints(anonymized_text: str, allowed_tags: Optional[Set[str]] = None) -> Dict[str, str]:
    """
    Déduit les rôles annoncés par le LLM dans le rapport anonymisé en analysant
    essentiellement la section Participants (tableaux ou listes).
    """
    roles: Dict[str, str] = {}
    if not anonymized_text:
        return roles

    for line in anonymized_text.splitlines():
        if "<PERSON_" not in line:
            continue
        stripped = line.strip()

        if "|" in stripped:
            cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
            if len(cells) >= 2:
                tag_cell = None
                category_cell = None
                for cell in cells:
                    if cell.upper().startswith("<PERSON_"):
                        normalized_tag = _normalize_tag_key(cell)
                        if allowed_tags and normalized_tag not in allowed_tags:
                            continue
                        tag_cell = normalized_tag
                        break
                if tag_cell:
                    for cell in cells:
                        normalized_role = _normalize_role_label(cell)
                        if normalized_role in {"Avocat gérante", "Delphine (avocat gérante)", "Collaborateur", "Client"}:
                            category_cell = normalized_role
                            break
                if tag_cell and category_cell:
                    roles[tag_cell] = category_cell
                    continue

        for match in re.finditer(r"<PERSON_\d+>", line, flags=re.IGNORECASE):
            tag = _normalize_tag_key(match.group(0))
            if allowed_tags and tag not in allowed_tags:
                continue
            tail = line[match.end():].lower()
            if "avocat" in tail and ("gerant" in tail or "gérant" in tail or "gérante" in tail or "delphine" in tail):
                roles[tag] = "Avocat gérante"
                continue
            if "collaborateur" in tail or "collaboratrice" in tail:
                roles[tag] = "Collaborateur"
                continue
            if "client" in tail:
                roles[tag] = "Client"

    return roles


def _refine_role_hints(
    roles: Dict[str, str],
    mapping: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, str]:
    if not roles:
        return roles

    refined: Dict[str, str] = {}
    lookup = _build_tag_lookup(mapping)

    for tag_key, role in roles.items():
        canonical = lookup.get(tag_key)
        if not canonical:
            continue
        canonical_lower = canonical.lower()
        count = counts.get(tag_key, 0)

        inferred_role = role
        for keyword, forced_role in _ROLE_KEYWORD_HINTS.items():
            if keyword in canonical_lower:
                inferred_role = forced_role
                break

        if inferred_role == "Client" and any(
            keyword in canonical_lower for keyword in ("marine", "marie", "delphine", "sylvie")
        ):
            inferred_role = "Collaborateur"

        if count <= 1 and inferred_role == "Client" and canonical_lower not in {"client"}:
            continue

        refined[tag_key] = inferred_role

    return refined


def _replace_first_occurrence(text: str, target: str, replacement: str) -> Tuple[str, bool]:
    pattern = re.compile(re.escape(target), flags=re.IGNORECASE)
    if not pattern.search(text):
        return text, False
    new_text = pattern.sub(replacement, text, count=1)
    return new_text, True


def _append_role_suffix(text: str, canonical: str, suffix: str) -> Tuple[str, bool]:
    pattern = re.compile(rf"({re.escape(canonical)})(?!\s*\()", flags=re.IGNORECASE)

    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{suffix}"

    new_text, count = pattern.subn(_repl, text, count=1)
    return new_text, bool(count)


def _strip_table_bullet_prefix(line: str) -> Tuple[str, bool]:
    """
    Retire un éventuel préfixe de puce (« - » ou « * ») placé juste avant un tableau Markdown.
    """
    match = re.match(r"^[ \t]*[-*+]\s*(\|.+)$", line)
    if not match:
        return line, False
    return match.group(1), True


def _format_table_block(block: List[str]) -> List[str]:
    rows: List[List[str]] = []
    max_cols = 0

    for idx, raw_line in enumerate(block):
        trimmed = raw_line.strip()
        if not trimmed or "|" not in trimmed:
            continue
        parts = [cell.strip() for cell in trimmed.split("|")]
        # retire les bordures vides dues aux pipes d'ouverture/fermeture mais
        # conserve les cellules vides internes (``||``) qui représentent une colonne vide
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]

        cells = parts

        if idx == 0:
            # en-tête : ignore les lignes d'underline accidentelles dans la dernière cellule uniquement
            if cells and all(ch in "-–—" for ch in cells[-1].replace(" ", "")):
                cells = cells[:-1]
        else:
            # saute les lignes composées uniquement de tirets (séparateurs)
            if all(not cell or all(ch in "-–—" for ch in cell.replace(" ", "")) for cell in cells):
                continue
        if len(cells) <= 1:
            return block  # pas un tableau : on ne modifie rien
        rows.append(cells)
        max_cols = max(max_cols, len(cells))

    if len(rows) < 2 or max_cols < 2:
        return block

    normalized: List[str] = []
    header = rows[0] + [""] * (max_cols - len(rows[0]))
    normalized.append("| " + " | ".join(header) + " |")
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    normalized.append(separator)

    for row in rows[1:]:
        padded = row + [""] * (max_cols - len(row))
        normalized.append("| " + " | ".join(padded) + " |")

    return normalized


def _normalize_markdown_tables(text: str) -> str:
    lines = text.splitlines()
    normalized_lines: List[str] = []
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        line, _ = _strip_table_bullet_prefix(raw_line)
        stripped = line.strip()
        if (
            "|" in line
            and not stripped.startswith("```")
            and not stripped.startswith("- ")
            and not stripped.startswith("* ")
            and not stripped.startswith("•")
        ):
            block: List[str] = []
            j = i
            last_had_pipe = False
            while j < len(lines):
                current_raw = lines[j]
                current, _ = _strip_table_bullet_prefix(current_raw)
                current_stripped = current.strip()
                if not current_stripped:
                    break
                if "|" in current:
                    block.append(current)
                    last_had_pipe = True
                    j += 1
                    continue
                if last_had_pipe and current_raw.startswith(" "):
                    block[-1] = block[-1].rstrip() + " " + current_stripped
                    j += 1
                    continue
                break
            if len(block) >= 2:
                normalized_lines.extend(_format_table_block(block))
                i = j
                continue
        normalized_lines.append(raw_line)
        i += 1

    return "\n".join(normalized_lines)


def _normalize_table_pipes(text: str) -> str:
    """
    Préserve les colonnes vides des tableaux Markdown : on évite d'écraser les
    « || » (cellule vide) en dehors des tableaux.
    """
    normalized: List[str] = []
    for raw in text.splitlines():
        stripped = raw.lstrip()
        candidate, _ = _strip_table_bullet_prefix(stripped)
        if candidate.lstrip().startswith("|") and candidate.count("|") >= 2:
            normalized.append(re.sub(r"\|\|", "| |", raw))
        else:
            normalized.append(raw)
    return "\n".join(normalized)


def _prune_participants_table(
    text: str,
    mapping: Dict[str, Any],
    roles: Dict[str, str],
    speaker_tags: Optional[Set[str]] = None,
    speaker_labels: Optional[Set[str]] = None,
) -> str:
    lookup = _build_tag_lookup(mapping)
    pseudo_lookup: Dict[str, str] = {}
    pseudo_map = mapping.get("pseudonym_map", {})
    if isinstance(pseudo_map, dict):
        for tag, pseudo in pseudo_map.items():
            normalized_tag = _normalize_tag_key(tag)
            if normalized_tag and isinstance(pseudo, str) and pseudo.strip():
                pseudo_lookup[normalized_tag] = pseudo.strip()
    entities = mapping.get("entities")
    if isinstance(entities, dict):
        for tag, info in entities.items():
            normalized_tag = _normalize_tag_key(tag)
            pseudo = (info or {}).get("pseudonym")
            if normalized_tag and isinstance(pseudo, str) and pseudo.strip():
                pseudo_lookup.setdefault(normalized_tag, pseudo.strip())

    allowed_keywords: Set[str] = set()
    allowed_tags: Set[str] = set()
    if speaker_tags:
        allowed_tags.update(_normalize_tag_key(tag) for tag in speaker_tags if tag)
    for tag_key in roles.keys():
        if not tag_key:
            continue
        allowed_tags.add(_normalize_tag_key(tag_key))

    if allowed_tags:
        for tag_key in allowed_tags:
            canonical = lookup.get(tag_key) or lookup.get(f"<{tag_key}>")
            pseudo_val = pseudo_lookup.get(tag_key) or pseudo_lookup.get(f"<{tag_key}>")
            if canonical:
                allowed_keywords.add(canonical.lower())
            if pseudo_val:
                allowed_keywords.add(pseudo_val.lower())
            allowed_keywords.add(tag_key.lower())
            allowed_keywords.add(f"<{tag_key.lower()}>")
    else:
        # fallback : autorise tous les pseudonymes / canoniques connus
        for value in list(pseudo_lookup.values()) + list(lookup.values()):
            if value:
                allowed_keywords.add(value.lower())

    if speaker_labels:
        for label in speaker_labels:
            if label:
                allowed_keywords.add(label.lower())

    # Inclus toujours les pseudonymes et noms connus même si les rôles sont absents
    if not allowed_keywords:
        for value in list(pseudo_lookup.values()) + list(lookup.values()):
            if value:
                allowed_keywords.add(value.lower())

    lines = text.splitlines()
    new_lines: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        if line.strip().lower().startswith("### participants"):
            i += 1
            table_block: List[str] = []
            while i < len(lines):
                current = lines[i]
                if current.strip().startswith("### "):
                    break
                if not current.strip() and table_block:
                    break
                table_block.append(current)
                i += 1

            filtered_block: List[str] = []
            data_rows: List[str] = []
            for idx, block_line in enumerate(table_block):
                stripped = block_line.strip()
                if not stripped.startswith("|"):
                    # conserve les lignes contextuelles éventuelles
                    filtered_block.append(block_line)
                    continue
                if idx in (0, 1):
                    # conserve l'en-tête et le séparateur de table
                    filtered_block.append(block_line)
                    continue
                cells = [cell.strip().lower() for cell in stripped.strip("|").split("|")]
                if all(not cell for cell in cells):
                    continue
                if all(ch == "-" for cell in cells for ch in cell.replace(" ", "")):
                    filtered_block.append(block_line)
                    continue
                data_rows.append(block_line)
                row_text = " ".join(cells)
                row_text_lower = row_text.lower()
                if allowed_keywords and any(keyword in row_text_lower for keyword in allowed_keywords):
                    filtered_block.append(block_line)
                elif not allowed_keywords:
                    filtered_block.append(block_line)

            if allowed_keywords:
                kept_data = [row for row in filtered_block if row.strip().startswith("|")][2:]
                if not kept_data and data_rows:
                    max_rows = len(allowed_tags) if allowed_tags else 1
                    fallback_rows = data_rows[:max_rows or 1]
                    for row in fallback_rows:
                        if row not in filtered_block:
                            filtered_block.append(row)
            new_lines.extend(filtered_block)
            if i < len(lines) and not lines[i].strip().startswith("### "):
                new_lines.append("")
            continue
        i += 1

    return "\n".join(new_lines)


def _append_missing_entities_section(text: str, mapping: Dict[str, Any]) -> str:
    grouped = _canonical_entities_by_label(mapping)
    if not grouped:
        return text

    normalized_text = text.casefold()
    missing: Dict[str, List[str]] = {}
    for label, names in grouped.items():
        for name in names:
            lowered = name.casefold()
            if lowered and lowered not in normalized_text:
                missing.setdefault(label, []).append(name)
    if not missing:
        return text

    section_lines: List[str] = ["", "### Référentiel des entités désanonymisées", ""]
    for label in sorted(missing.keys()):
        section_lines.append(f"**{label.title()}**")
        for name in missing[label]:
            section_lines.append(f"- {name}")
        section_lines.append("")

    base_text = text.rstrip() + "\n\n"
    return base_text + "\n".join(section_lines).rstrip() + "\n"


def _rewrite_roles(
    deanonymized_text: str,
    mapping: Dict[str, Any],
    roles: Dict[str, str],
) -> str:
    if not roles:
        return deanonymized_text

    lookup = _build_tag_lookup(mapping)
    alias_source = mapping.get("alias_lookup", {})
    alias_lookup: Dict[str, str] = {}
    if isinstance(alias_source, dict):
        for tag, alias in alias_source.items():
            if not isinstance(tag, str) or not isinstance(alias, str):
                continue
            normalized_tag = _normalize_tag_key(tag)
            if not normalized_tag:
                continue
            alias_lookup[normalized_tag] = alias.strip()
    text = deanonymized_text

    for tag, role in roles.items():
        normalized_tag = _normalize_tag_key(tag)
        canonical = lookup.get(tag)
        if not canonical:
            continue

        normalized_role = role.lower()
        preferred_display = alias_lookup.get(normalized_tag, canonical)
        if "delphine" in normalized_role:
            alias_note = ""
            if canonical and preferred_display and preferred_display.lower() != canonical.lower():
                alias_note = f'; transcription : "{canonical}"'
            detailed_label = f'{preferred_display or "Delphine"} (avocat gérante{alias_note})'
            text, replaced = _replace_first_occurrence(text, canonical, detailed_label)
            if replaced:
                canonical_pattern = re.compile(rf"\b{re.escape(canonical)}\b", flags=re.IGNORECASE)

                def _delphine_repl(match: re.Match[str]) -> str:
                    before = match.string[max(0, match.start() - 1) : match.start()]
                    after = match.string[match.end() : match.end() + 1]
                    if before in {'"', "'"} and after in {'"', "'"}:
                        return match.group(0)
                    if before == "(":
                        return match.group(0)
                    return preferred_display or "Delphine"

                text = canonical_pattern.sub(_delphine_repl, text)
            else:
                fallback_display = preferred_display or "Delphine"
                if fallback_display not in text:
                    note = f'; transcription : "{canonical}"' if canonical and (fallback_display.lower() != canonical.lower()) else ""
                    text += f'\n\n{fallback_display} (avocat gérante{note})'
            continue

        suffix = f" ({role})"
        role_pattern = re.compile(
            rf"{re.escape(canonical)}\s*\(([^\)]*{re.escape(role)}[^\)]*)\)",
            flags=re.IGNORECASE,
        )
        if role_pattern.search(text):
            continue
        text, appended = _append_role_suffix(text, canonical, suffix)
        if not appended:
            fallback_pattern = re.compile(rf"\b{re.escape(canonical)}\b", flags=re.IGNORECASE)
            text = fallback_pattern.sub(f"{canonical}{suffix}", text, count=1)

    return text


_INVENTED_TAG_RE = re.compile(r"<\s*([A-Za-z]+(?:_[A-Za-z0-9]+)?)\s*>")
_ALLOWED_HTML_TAGS = {"BR"}


def _drop_unknown_tags(text: str, known_tags: Set[str]) -> str:
    if not known_tags:
        return text

    def _repl(match: re.Match[str]) -> str:
        tag = _normalize_tag_key(match.group(1))
        if tag in _ALLOWED_HTML_TAGS:
            return match.group(0)
        return "" if tag not in known_tags else match.group(0)

    return _INVENTED_TAG_RE.sub(_repl, text)


def _normalize_bullets_outside_tables(text: str) -> str:
    """
    Force a newline before list bullets, but skip Markdown table rows to avoid
    breaking column alignment.
    """
    normalized_lines: List[str] = []
    for line in text.splitlines():
        if "|" in line:
            normalized_lines.append(line)
            continue
        normalized_lines.append(re.sub(r"(\S)(-\s+\*\*|-\s+)", r"\1\n\2", line))
    return "\n".join(normalized_lines)


def _strip_preamble(text: str) -> str:
    """
    Supprime tout préambule éventuel avant la première section (### ...).
    """
    first_header = text.find("###")
    if first_header == -1:
        return text
    return text[first_header:].lstrip()


def _split_compound_bullets(text: str) -> str:
    """
    Scinde les lignes de puces qui enchaînent plusieurs éléments sur la même
    ligne (\"- **Item1** ... - Item2 ...\") en puces distinctes.
    """
    split_lines: List[str] = []
    for line in text.splitlines():
        if "|" in line:
            split_lines.append(line)
            continue
        expanded = re.sub(r"\s+-\s+(?=[*-])", "\n- ", line)
        split_lines.extend(expanded.splitlines())
    return "\n".join(split_lines)


def _normalize_inline_numbering(text: str) -> str:
    """
    Forcer les items numérotés collés (\"1. ... 2. ...\") à revenir sur des
    lignes distinctes tout en conservant la numérotation.
    """
    normalized: List[str] = []
    for raw_line in text.splitlines():
        if "|" in raw_line:
            normalized.append(raw_line)
            continue
        if raw_line.lstrip().startswith("#"):
            normalized.append(raw_line.rstrip())
            continue

        line = raw_line.rstrip()
        if not line:
            normalized.append("")
            continue

        split = re.sub(r"\s+(?=\d+\.\s)", "\n", line)
        parts = split.splitlines() or [split]
        for part in parts:
            stripped = part.strip()
            if not stripped:
                normalized.append("")
                continue
            match = re.match(r"^(\d+)\.\s+(.*)", stripped)
            if match:
                normalized.append(f"{match.group(1)}. {match.group(2).strip()}")
            else:
                normalized.append(stripped)
    return "\n".join(normalized)


def _normalize_section_headers(text: str) -> str:
    """
    Remet chaque section (### ...) sur sa propre ligne et insère des sauts de
    ligne pour éviter les titres collés aux paragraphes ou listes.
    """
    cleaned = re.sub(r"\s*--\s*###", "\n\n###", text)
    cleaned = re.sub(r"\s*-\s*###", "\n###", cleaned)
    cleaned = re.sub(r"(?<!\n)###", r"\n###", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    normalized: List[str] = []
    last_blank = False
    for raw in cleaned.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("###"):
            header, trailing = _split_section_header_line(line.strip())
            if normalized and normalized[-1].strip():
                normalized.append("")
            normalized.append(header)
            if trailing:
                trailing = trailing.lstrip("-–—: ").strip()
                if trailing and not trailing.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                    trailing = f"- {trailing}"
                if trailing:
                    normalized.append("")
                    normalized.append(trailing)
            last_blank = False
            continue
        if not line.strip():
            if last_blank:
                continue
            last_blank = True
            normalized.append("")
            continue
        last_blank = False
        normalized.append(line)

    return "\n".join(normalized).strip() + "\n"


def _number_main_sections(text: str) -> str:
    """
    Préfixe systématiquement les sections principales par un numéro ordonné (1..N)
    pour garantir une arborescence stable dans le rendu Markdown.
    """
    normalized: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^####", stripped):
            normalized.append(line.rstrip())
            continue

        # Titres déjà au bon format
        if re.match(r"^###", stripped):
            header = re.sub(r"^#+\s*", "", stripped)
            header = re.sub(r"^\d+[.)]\s*", "", header)
        else:
            header = re.sub(r"^\d+[.)]\s*", "", stripped)

        header_key = _normalize_title_key(header)
        matched_title = None
        for title in _MAIN_SECTIONS:
            if _normalize_title_key(title) == header_key or header_key.startswith(
                _normalize_title_key(title)
            ):
                matched_title = title
                break

        if matched_title:
            section_idx = _MAIN_SECTIONS.index(matched_title) + 1
            normalized.append(f"### {section_idx}. {matched_title}")
            continue

        normalized.append(line.rstrip())
    return "\n".join(normalized)


def _drop_lonely_markers(text: str) -> str:
    """
    Élimine les lignes résiduelles ne contenant qu'un « # » ou « | ».
    """
    cleaned: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {"#", "|"} or re.match(r"^#{2,6}\s*$", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _normalize_risk_opp_headers(text: str) -> str:
    """
    Harmonise les titres Risques / Opportunités sur le même niveau de titre.
    """
    normalized: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^#{3,6}\s+risques$", stripped, flags=re.IGNORECASE):
            normalized.append("#### Risques")
            continue
        if re.match(r"^#{3,6}\s+opportunités$", stripped, flags=re.IGNORECASE):
            normalized.append("#### Opportunités")
            continue
        normalized.append(line)
    return "\n".join(normalized)


def _split_numbered_items(block: str) -> List[str]:
    pattern = re.compile(r"\d+[.)]\s+")
    matches = list(pattern.finditer(block))
    if not matches:
        return []

    items: List[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
        candidate = block[start:end].strip(" \n-:•")
        if candidate:
            items.append(candidate)
    return items


def _split_risk_opp_content(block: str) -> Tuple[List[str], List[str]]:
    """
    Sépare le contenu de la section Risques/Opportunités en deux listes.
    Gère les formats linéaires «Risques : 1. ...» ou déjà en puces.
    """
    if not block:
        return [], []

    lower = block.lower()
    pos_r = lower.find("risque")
    pos_o = lower.find("opportun")

    risks_raw = ""
    opps_raw = ""
    if pos_r != -1 and (pos_o == -1 or pos_r < pos_o):
        risks_raw = block[pos_r : pos_o if pos_o != -1 else len(block)]
        opps_raw = block[pos_o:] if pos_o != -1 else ""
    elif pos_o != -1 and (pos_r == -1 or pos_o < pos_r):
        opps_raw = block[pos_o : pos_r if pos_r != -1 else len(block)]
        risks_raw = block[pos_r:] if pos_r != -1 else ""
    else:
        risks_raw = block

    def _clean(text: str) -> List[str]:
        cleaned = re.sub(r"(?i)^risques?\s*[:\-]\s*", "", text).strip()
        cleaned = re.sub(r"(?i)^opportunités?\s*[:\-]\s*", "", cleaned).strip()
        if not cleaned:
            return []
        bullet_lines = [
            line.strip(" -•").strip()
            for line in cleaned.splitlines()
            if line.strip().startswith(("-", "•"))
        ]
        if bullet_lines:
            return bullet_lines
        numbered = _split_numbered_items(cleaned)
        if numbered:
            return numbered
        sentences = [
            sent.strip(" -•")
            for sent in re.split(r"(?<=[.!?])\s+", cleaned)
            if sent.strip()
        ]
        return sentences

    return _clean(risks_raw), _clean(opps_raw)


def _restructure_risk_opp_section(text: str) -> str:
    """
    Reformate la section Risques/Opportunités en listes distinctes pour éviter
    les blocs collés sur une seule ligne.
    """

    def _repl(match: re.Match[str]) -> str:
        header = match.group(1).strip() + "\n"
        content = match.group(2)
        tail = match.group(3)

        risks, opps = _split_risk_opp_content(content)
        section_lines: List[str] = [header.rstrip(), ""]
        if risks:
            section_lines.append("**Risques :**")
            section_lines.extend(f"- {item}" for item in risks)
            section_lines.append("")
        if opps:
            if section_lines and section_lines[-1].strip():
                section_lines.append("")
            section_lines.append("**Opportunités :**")
            section_lines.extend(f"- {item}" for item in opps)
            section_lines.append("")
        if not risks and not opps:
            section_lines.append(content.strip())
            section_lines.append("")

        rebuilt = "\n".join(section_lines).rstrip() + "\n"
        if tail.startswith("\n###"):
            return rebuilt + "\n" + tail.lstrip("\n")
        return rebuilt + tail

    pattern = re.compile(
        r"(###\s*\d*[.)]?\s*ANALYSE[S]?\s+DES\s+RISQUES\s+ET\s+OPPORTUNIT[ÉE]S?\s*\n)(.*?)(\n###\s|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub(_repl, text)


def _ensure_risk_opp_spacing(text: str) -> str:
    """
    Garantit une ligne vide entre le bloc Risques et le bloc Opportunités
    sans réécrire le contenu fourni par le LLM.
    """
    lines = text.splitlines()
    out: List[str] = []
    for idx, line in enumerate(lines):
        out.append(line.rstrip())
        if (
            line.strip().lower().startswith("**risques")
            and idx + 1 < len(lines)
            and lines[idx + 1].strip()
            and not lines[idx + 1].strip().startswith("**opportun")
        ):
            continue
        if line.strip().lower().startswith("**risques") and idx + 1 < len(lines):
            next_line = lines[idx + 1].strip().lower()
            if next_line.startswith("**opportun"):
                out.append("")
    return "\n".join(out)


def _strip_header_bullets(text: str) -> str:
    """
    Supprime les lignes composées uniquement d'un « - » juste avant un titre ###,
    qui provoquent un artefact visuel dans le rendu Markdown/PDF.
    """
    lines = text.splitlines()
    cleaned: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "-" and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("###"):
            i += 1
            continue
        cleaned.append(line)
        i += 1
    return "\n".join(cleaned)


def _ensure_table_spacing(text: str) -> str:
    """
    Ajoute des lignes vides avant et après les blocs de tableaux Markdown
    pour éviter les rendus cassés dans les exports pandoc/LaTeX.
    """
    lines = text.splitlines()
    out: List[str] = []
    in_table = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        is_table_line = stripped.startswith("|")

        if is_table_line:
            if not in_table:
                if out and out[-1].strip():
                    out.append("")
                in_table = True
            out.append(line.rstrip())
            continue

        if in_table:
            if stripped and out and out[-1].strip():
                out.append("")
            in_table = False

        out.append(line.rstrip())

    return "\n".join(out)


def _restore_executive_summary_spacing(text: str) -> str:
    """
    Restaure des sauts de ligne lisibles dans le résumé exécutif en ajoutant
    une ligne vide après le titre et entre paragraphes/listes clés.
    """
    pattern = re.compile(
        r"(###\s*\d*[.)]?\s*RÉSUMÉ EXÉCUTIF\s*\n)(.*?)(\n###\s|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _repl(match: re.Match[str]) -> str:
        header = match.group(1).strip()
        content = match.group(2).strip("\n")
        tail = match.group(3)

        lines = content.splitlines()
        restored: List[str] = []
        last_nonempty = False
        for line in lines:
            stripped = line.strip()
            starts_list = stripped.startswith("- ")
            is_subheading = stripped.lower().startswith(
                (
                    "les enjeux majeurs",
                    "décisions clés",
                    "risques identifiés",
                    "prochaines étapes",
                )
            )
            if restored and stripped and (starts_list or is_subheading or last_nonempty):
                if restored[-1].strip():
                    restored.append("")
            restored.append(line.rstrip())
            last_nonempty = bool(stripped)

        block = "\n".join(restored).strip()
        rebuilt = f"{header}\n\n{block}\n"
        if tail.startswith("\n###"):
            rebuilt += "\n" + tail.lstrip("\n")
        else:
            rebuilt += tail
        return rebuilt

    return pattern.sub(_repl, text)


def _normalize_points_sections(text: str) -> str:
    """
    Harmonise les listes des sections POINTS POSITIFS / POINTS DE FRICTION :
    les items principaux restent en puces, les détails passent en sous-puces.
    """
    result: List[str] = []
    in_points_block = False

    def _is_points_header(line: str) -> bool:
        stripped = line.strip().lower()
        return stripped.startswith("### points positifs") or stripped.startswith(
            "### points de friction"
        )

    for line in text.splitlines():
        stripped = line.lstrip()
        if _is_points_header(line):
            in_points_block = True
            result.append(line.strip())
            continue
        if stripped.startswith("### ") and not _is_points_header(line):
            in_points_block = False
            result.append(line)
            continue
        if in_points_block and stripped.startswith("- "):
            if stripped.startswith("- **"):
                result.append(f"- {stripped[2:].lstrip()}")
            else:
                result.append(f"  - {stripped[2:].lstrip()}")
            continue
        result.append(line)

    return "\n".join(result)


def _normalize_actions_section(text: str) -> str:
    """
    Nettoie la section ACTIONS pour éviter les puces parasites avant le tableau
    (ex. « - À MENER ») qui cassent le rendu Markdown/PDF.
    """
    section_pattern = re.compile(
        r"(###\s*\d*[.)]?\s*ACTIONS[^\n]*\n)(.*?)(\n###\s|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _repl(match: re.Match[str]) -> str:
        header = match.group(1).strip()
        content = match.group(2)
        tail = match.group(3)

        cleaned_lines: List[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            # supprime les puces « - À MENER ... » (même si suivi de texte) qui cassent le tableau
            if re.match(r"^[-*]\s*[àa]?\s*mener\b", stripped, flags=re.IGNORECASE):
                continue
            cleaned_lines.append(line.rstrip())

        # assure une ligne blanche avant un tableau pour éviter un rendu imbriqué
        if cleaned_lines and cleaned_lines[0].strip().startswith("|"):
            cleaned_lines.insert(0, "")

        block = "\n".join([header, *cleaned_lines]).rstrip()
        if tail.startswith("\n###"):
            block += "\n" + tail.lstrip("\n")
        return block

    return section_pattern.sub(_repl, text)


def _format_decisions_as_bullets(text: str) -> str:
    """
    Convertit la section DÉCISIONS en liste à puces avec sous-points indentés
    (Décision / Contexte / Intervenants / Citation) pour faciliter la lecture.
    Supprime la numérotation éventuelle laissée par le LLM.
    """
    section_pattern = re.compile(
        r"(###\s*\d*[.)]?\s*DÉCISIONS\s*\n)(.*?)(\n###\s|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    field_patterns = {
        "decision": re.compile(r"^d[ée]cision\s*:?\s*(.*)", flags=re.IGNORECASE),
        "contexte": re.compile(r"^contexte\s*:?\s*(.*)", flags=re.IGNORECASE),
        "intervenants": re.compile(r"^intervenants?\s*:?\s*(.*)", flags=re.IGNORECASE),
        "citation": re.compile(r"^citation\s*:?\s*(.*)", flags=re.IGNORECASE),
    }

    def _clean(line: str) -> str:
        cleaned = line.strip()
        cleaned = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", cleaned)
        return cleaned.strip(" -–—")

    def _format_item(raw_lines: List[str]) -> List[str]:
        if not raw_lines:
            return []

        header = _clean(raw_lines[0])
        fields: Dict[str, List[str]] = {k: [] for k in field_patterns.keys()}
        extras: List[str] = []

        for raw in raw_lines[1:]:
            cleaned = _clean(raw)
            if not cleaned:
                continue
            matched = False
            for key, pattern in field_patterns.items():
                match = pattern.match(cleaned)
                if match:
                    value = (match.group(1) or "").strip(" :-")
                    if value:
                        fields[key].append(value)
                    matched = True
                    break
            if not matched:
                extras.append(cleaned)

        if header:
            if not fields["decision"]:
                fields["decision"].append(header)
            elif header not in fields["decision"]:
                fields["decision"].insert(0, header)

        if extras and not fields["contexte"]:
            fields["contexte"].append(" ".join(extras))

        def _val(key: str, fallback: Optional[str] = None) -> str:
            joined = " ".join(fields[key]).strip()
            if joined:
                return joined
            if fallback:
                return fallback
            return "Non précisé"

        decision_title = header or "Décision"
        decision_text = _val("decision", decision_title)
        context_text = _val("contexte", decision_title if extras else None)
        intervenants_text = _val("intervenants")
        citation_text = _val("citation")

        return [
            f"- {decision_title}",
            f"  - Décision : {decision_text}",
            f"  - Contexte : {context_text}",
            f"  - Intervenants : {intervenants_text}",
            f"  - Citation : {citation_text}",
        ]

    def _extract_items(block: str) -> List[List[str]]:
        items: List[List[str]] = []
        current: List[str] = []

        def _is_field_line(raw: str) -> bool:
            normalized = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", raw).strip()
            return bool(
                re.match(
                    r"(?i)^(d[ée]cision|contexte|intervenants?|citation)\b",
                    normalized,
                )
            )

        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                if current:
                    current.append("")
                continue

            is_field = _is_field_line(stripped)
            is_bullet = bool(re.match(r"^(?:[-*]|\d+[.)])\s+", stripped))

            if is_bullet and is_field:
                # Conserve ces lignes comme sous-éléments de l'item courant
                if not current:
                    current = [stripped]
                else:
                    current.append(stripped)
                continue

            if is_bullet:
                if current:
                    items.append(current)
                current = [stripped]
            else:
                current.append(stripped)

        if current:
            items.append(current)
        return items

    def _repl(match: re.Match[str]) -> str:
        header = match.group(1).strip()
        content = match.group(2)
        tail = match.group(3)

        items = _extract_items(content)
        if not items:
            return match.group(0)

        formatted: List[str] = [header, ""]
        for idx, item in enumerate(items):
            formatted.extend(_format_item(item))
            if idx < len(items) - 1:
                formatted.append("")

        block = "\n".join(line.rstrip() for line in formatted if line is not None).rstrip()
        if tail.startswith("\n###"):
            block += "\n" + tail.lstrip("\n")
        return block

    return section_pattern.sub(_repl, text)


def _indent_numbered_children(text: str) -> str:
    """
    Indente les sous-puces qui suivent un item numéroté (1., 2., etc.) pour
    rendre la hiérarchie lisible dans les exports Markdown/Docx/PDF.
    """
    result: List[str] = []
    in_numbered_block = False

    for line in text.splitlines():
        stripped = line.lstrip()

        if "|" in line or stripped.startswith("###"):
            in_numbered_block = False
            result.append(line.rstrip())
            continue

        if not stripped:
            in_numbered_block = False
            result.append("")
            continue

        if re.match(r"^\d+\.\s", stripped):
            in_numbered_block = True
            result.append(stripped)
            continue

        if in_numbered_block and stripped.startswith("- "):
            result.append(f"  {stripped}")
            continue

        result.append(line.rstrip())

    return "\n".join(result)


def _split_residual_inline_items(text: str) -> str:
    """
    Dernier filet de sécurité : éclate les items numérotés ou puces restés sur
    la même ligne pour garantir une arborescence lisible.
    """
    lines: List[str] = []
    for raw_line in text.splitlines():
        if "|" in raw_line:
            lines.append(raw_line.rstrip())
            continue
        if raw_line.lstrip().startswith("#"):
            lines.append(raw_line.rstrip())
            continue

        working = re.sub(r"\s+(?=\d+\.\s)", "\n", raw_line)
        working = re.sub(r"\s+-\s+(?=[*-])", "\n- ", working)
        parts = working.splitlines()
        if not parts:
            lines.append("")
            continue
        lines.extend(part.rstrip() for part in parts)

    return "\n".join(lines)


def _strip_trailing_markers(text: str) -> str:
    """
    Supprime les marqueurs résiduels type \"--\" laissés en fin de bloc.
    """
    return re.sub(r"\s*--\s*$", "", text, flags=re.MULTILINE)


def _compact_numbered_lists(text: str) -> str:
    """
    Supprime les lignes vides au milieu d'une liste numérotée pour éviter que
    pandoc ne crée plusieurs listes séparées (et donc reparties à 1).
    """
    lines = text.splitlines()
    out: List[str] = []
    in_numbered = False

    for raw in lines:
        stripped = raw.strip()
        is_numbered = bool(re.match(r"^\d+[.)]\s", stripped))
        if not stripped and in_numbered:
            # on saute les lignes vides entre deux items
            continue
        out.append(raw.rstrip())
        in_numbered = is_numbered
        if not stripped:
            in_numbered = False

    return "\n".join(out)


def _renumber_markdown_lists(text: str) -> str:
    """
    Renumérote proprement les listes numérotées Markdown pour éviter le
    ``1.`` répété sur chaque ligne après les réécritures.
    """
    lines = text.splitlines()
    out: List[str] = []
    stack: List[Tuple[str, int]] = []

    def _reset():
        stack.clear()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.lstrip()
        if "|" in line or stripped.startswith("#"):
            _reset()
            out.append(line)
            continue

        match = re.match(r"^(\s*)(\d+)[.)]\s+(.*)", line)
        if match:
            indent = match.group(1)
            content = match.group(3)

            # Ajuste la pile en fonction de l'indentation
            while stack and len(stack[-1][0]) > len(indent):
                stack.pop()
            if stack and stack[-1][0] == indent:
                stack[-1] = (indent, stack[-1][1] + 1)
            else:
                stack.append((indent, 1))

            out.append(f"{indent}{stack[-1][1]}. {content}")
            continue

        if not stripped or stripped.startswith(("-", "*")):
            _reset()
        out.append(line)

    return "\n".join(out)


def _fix_inline_section_tables(text: str) -> str:
    """
    Extrait les en-têtes de section écrites à l'intérieur d'une ligne de tableau
    (ex.: \"| ### PARTICIPANTS | ...\") et les remet sur leur propre ligne.
    """
    lines = text.splitlines()
    fixed: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped == "|":
            fixed.append("")
            continue

        for pattern in (r"^\|\s*###\s+([^|]+?)\s*\|\s*(.+)$", r"^###\s+([^|]+?)\s*\|\s*(.+)$"):
            match = re.match(pattern, line)
            if match:
                section = match.group(1).strip()
                rest = match.group(2).strip()
                fixed.append(f"### {section}")
                fixed.append("")
                if not rest.startswith("|"):
                    rest = f"| {rest}"
                fixed.append(rest)
                break
        else:
            fixed.append(line)

    return "\n".join(fixed)


def _structure_executive_summary_content(block: str) -> str:
    """
    Rend le résumé exécutif plus lisible : si le bloc ne contient pas déjà de
    puces ou de tableaux, produit un paragraphe rédigé suivi éventuellement
    d'une courte liste de points clés.
    """
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return ""

    has_list = any(
        line.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.")) or "|" in line
        for line in lines
    )
    if has_list:
        return "\n".join(lines) + "\n"

    sentence_split = re.split(r"(?<=[.!?])\s+(?=[A-ZÉÈÀÂÎÔÙÇ])", " ".join(lines))
    sentences = [sent.strip(" \n-") for sent in sentence_split if sent.strip()]
    if len(sentences) <= 1:
        return "\n\n".join(lines) + "\n"

    lead_count = 2 if len(sentences) > 3 else len(sentences)
    lead_paragraph = " ".join(sentences[:lead_count])
    tail = sentences[lead_count:]

    assembled: List[str] = [lead_paragraph]
    if tail:
        assembled.append("")
        assembled.append("Points clés :")
        assembled.extend(f"- {sent}" for sent in tail)

    return "\n".join(assembled).rstrip() + "\n"


def _format_executive_summary(text: str) -> str:
    """
    Structure le résumé exécutif pour favoriser un rendu aéré (liste de points).
    """

    def _repl(match: re.Match[str]) -> str:
        header = match.group(1).strip() + "\n"
        content = match.group(2)
        formatted = _structure_executive_summary_content(content)
        tail = match.group(3)
        if tail.startswith("\n###"):
            tail = tail[1:]
        return header + formatted + tail

    pattern = re.compile(
        r"(###\s*\d*[.)]?\s*RÉSUMÉ EXÉCUTIF\s*\n)(.*?)(\n###\s|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub(_repl, text)


def _slugify_title(title: str) -> str:
    normalized = _normalize_title_key(title)
    slug = re.sub(r"[^A-Z0-9]+", "-", normalized).strip("-").lower()
    return slug or "section"


def _wrap_sections_with_cards(text: str) -> str:
    """
    Encapsule chaque section (### ...) dans un bloc stylable (fenced div).
    """
    lines = text.splitlines()
    wrapped: List[str] = []
    open_block = False

    def _close_block() -> None:
        nonlocal open_block
        if open_block:
            wrapped.append(":::")
            wrapped.append("")
            open_block = False

    for line in lines:
        if line.strip().startswith("###"):
            _close_block()
            header = re.sub(r"^#+\s*", "", line).strip()
            header = re.sub(r"^\d+[.)]\s*", "", header)
            slug = _slugify_title(header)
            wrapped.append(f"::: section-card section-{slug}")
            wrapped.append("")
            open_block = True
        wrapped.append(line.rstrip())

    _close_block()
    return "\n".join(wrapped).strip() + "\n"


def _materialize_section_cards(markdown_text: str) -> str:
    """
    Convertit les blocs ::: section-card ... en balises <div> pour éviter
    l'affichage brut des marqueurs lorsque les extensions Markdown ne sont
    pas appliquées par le moteur (PDF/HTML).
    """
    lines = markdown_text.splitlines()
    rendered: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(":::"):
            parts = stripped.split()
            classes = parts[1:]
            if classes:
                rendered.append(f'<div class="{" ".join(classes)}">')
            else:
                rendered.append("</div>")
            continue
        rendered.append(line)

    return "\n".join(rendered)


def _strip_speaker_placeholders(text: str) -> str:
    """
    Supprime les mentions SPEAKER_# / SPK# résiduelles pour éviter leur affichage
    dans les rapports lorsque le contexte locuteurs est fourni.
    """
    cleaned = re.sub(r"\s*\(?\bSPEAKER_\d+\b\)?", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\bSPK\d+\b", "", cleaned, flags=re.IGNORECASE)
    # Nettoie les espaces laissés par la suppression.
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned


def _list_system_fonts() -> Set[str]:
    try:
        output = subprocess.check_output(
            ["fc-list", "--format=%{family}\\n"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return set()

    fonts: Set[str] = set()
    for raw_line in output.splitlines():
        if not raw_line:
            continue
        for family in raw_line.split(","):
            cleaned = family.strip()
            if cleaned:
                fonts.add(cleaned)
    return fonts


def _pick_preferred_font() -> Optional[str]:
    fonts = _list_system_fonts()
    if not fonts:
        return None
    lower_fonts = {font.lower(): font for font in fonts}
    for preferred in _PREFERRED_SANS_FONTS:
        if preferred.lower() in lower_fonts:
            return lower_fonts[preferred.lower()]
        for font in fonts:
            if font.lower().startswith(preferred.lower()):
                return font
    return None


def _convert_markdown_with_pandoc(markdown_text: str, to: str, out_path: Path) -> None:
    """
    Convert markdown text to the requested ``to`` format using pypandoc.
    """
    if not _HAS_PYPANDOC:
        raise RuntimeError(
            "pypandoc is required to export meeting reports "
            f"({to}) but is unavailable."
        ) from _PYPANDOC_IMPORT_ERROR

    extra_args = ["--standalone", "--wrap=none"]
    format_spec = "gfm" if to == "pdf" else _PANDOC_MD_FORMAT

    if to == "pdf":
        pdf_engine = os.getenv("PYPANDOC_PDF_ENGINE") or "xelatex"
        extra_args.extend(["--pdf-engine", pdf_engine])
        extra_args.extend(["--variable", "geometry:margin=1.5cm"])
        font_args: List[str] = []
        preferred_font = _pick_preferred_font()
        if preferred_font:
            font_args = [
                "--variable",
                f"mainfont={preferred_font}",
                "--variable",
                f"sansfont={preferred_font}",
            ]
            extra_args.extend(font_args)

        try:
            pypandoc.convert_text(  # type: ignore[call-arg]
                markdown_text,
                to=to,
                format=format_spec,
                outputfile=str(out_path),
                extra_args=extra_args,
            )
            return
        except RuntimeError:
            if os.getenv("PYPANDOC_PDF_ENGINE"):
                raise

            if pdf_engine == "xelatex":
                fallback_args = list(extra_args)
                if font_args:
                    fallback_args = fallback_args[:-len(font_args)]
                for idx, arg in enumerate(fallback_args):
                    if arg == "--pdf-engine" and idx + 1 < len(fallback_args):
                        fallback_args[idx + 1] = "pdflatex"
                        break
                pypandoc.convert_text(  # type: ignore[call-arg]
                    markdown_text,
                    to=to,
                    format=format_spec,
                    outputfile=str(out_path),
                    extra_args=fallback_args,
                )
                return
            raise

    pypandoc.convert_text(  # type: ignore[call-arg]
        markdown_text,
        to=to,
        format=format_spec,
        outputfile=str(out_path),
        extra_args=extra_args,
    )


def _get_report_assets() -> Tuple[Path, Path]:
    """
    Return the HTML template and CSS paths for the meeting report.
    """
    if not _REPORT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Meeting report template missing: {_REPORT_TEMPLATE_PATH}")
    if not _REPORT_CSS_PATH.exists():
        raise FileNotFoundError(f"Meeting report CSS missing: {_REPORT_CSS_PATH}")
    return _REPORT_TEMPLATE_PATH, _REPORT_CSS_PATH


def _build_html_report(markdown_text: str, *, title: Optional[str] = None) -> str:
    if not _HAS_PYPANDOC:
        raise RuntimeError(
            "pypandoc is required to render HTML reports"
        ) from _PYPANDOC_IMPORT_ERROR

    template_path, _ = _get_report_assets()
    extra_args = ["--standalone", "--template", str(template_path)]
    today = datetime.now().strftime("%d/%m/%Y")
    extra_args.extend(["--metadata", f"date={today}"])
    if title:
        extra_args.extend(["--metadata", f"title={title}"])

    return pypandoc.convert_text(  # type: ignore[call-arg]
        markdown_text,
        to="html",
        format=_PANDOC_MD_FORMAT,
        extra_args=extra_args,
    )


def _render_pdf_report(markdown_text: str, out_path: Path, *, title: Optional[str] = None) -> None:
    if not _HAS_WEASYPRINT:
        raise RuntimeError(
            "weasyprint is required to export meeting reports to PDF"
        ) from _WEASYPRINT_IMPORT_ERROR

    template_path, css_path = _get_report_assets()
    html_report = _build_html_report(markdown_text, title=title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_report, base_url=str(template_path.parent)).write_pdf(
        target=str(out_path),
        stylesheets=[CSS(filename=str(css_path))],
    )


def generate_meeting_report(
    anonymized_txt_path: Path,
    mapping_json_path: Path,
    prompts_json_path: Path,
    prompt_key: str = "meeting_analysis",
    out_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    speaker_context: Optional[str] = None,
):
    """
    1) Lit la transcription anonymisée (txt)
    2) Lance Mistral Large avec le prompt JSON (key configurable)
    3) Désanonymise le rapport LLM via mapping
    4) Exporte TXT/Markdown, DOCX via pypandoc et PDF via template HTML/CSS (WeasyPrint)

    ``speaker_context`` doit déjà être anonymisé (pseudonymes) pour éviter toute fuite
    vers l'API ; il est injecté en préambule du prompt pour aider l'attribution des rôles.
    """
    ok, failure_reason = _check_mistral_access()
    if not ok:
        message = failure_reason or "API Mistral indisponible"
        print(f"⚠️ Meeting report skipped: {message}")
        return _empty_report_outputs(message)

    out_dir = out_dir or anonymized_txt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) charge prompt + texte
    prompt = mistral_client.load_prompts(str(prompts_json_path), key=prompt_key)
    anon_text = Path(anonymized_txt_path).read_text(encoding="utf-8")
    mapping_raw = json.loads(Path(mapping_json_path).read_text(encoding="utf-8"))
    anon_text, mapping_raw = _rationalize_person_tags(anon_text, mapping_raw)
    person_counts = _count_person_tags(anon_text)
    speaker_labels = _extract_speaker_labels(anon_text)
    speaker_tags = _select_speaker_tags(
        mapping_raw,
        person_counts,
        max_speakers=len(speaker_labels) if speaker_labels else 0,
    )
    relevant_tags = _select_relevant_person_tags(mapping_raw, person_counts)

    # 2) Mistral Large
    pseudonym_hint = _build_pseudonym_hint(mapping_raw)
    speakers_hint = _build_speakers_hint(speaker_labels)
    hint_block = f"{pseudonym_hint}\n" if pseudonym_hint else ""
    speakers_block = f"{speakers_hint}" if speakers_hint else ""
    context_block = _format_speaker_context(speaker_context)
    user_payload = prompt.user_prefix + hint_block + speakers_block + context_block + anon_text
    analysis_anonymized = mistral_client.chat_complete(
        model=prompt.model,
        system=prompt.system,
        user_text=user_payload,
        temperature=0.1,
    )

    analysis_anonymized = _normalize_llm_placeholders(analysis_anonymized)
    if speaker_context:
        analysis_anonymized = _strip_speaker_placeholders(analysis_anonymized)
    role_hints = _extract_role_hints(analysis_anonymized, allowed_tags=relevant_tags)

    # 3) Désanonymisation
    mapping = _ensure_legacy_mapping(mapping_raw)
    role_hints = _refine_role_hints(role_hints, mapping, person_counts)
    _log_name_resolution(mapping)
    analysis_deanonymized = deanonymize_text(analysis_anonymized, mapping, restore="canonical")
    if speaker_context:
        analysis_deanonymized = _strip_speaker_placeholders(analysis_deanonymized)
    analysis_deanonymized = _rewrite_roles(analysis_deanonymized, mapping, role_hints)
    analysis_deanonymized = _drop_unknown_tags(analysis_deanonymized, _collect_known_tags(mapping))

    # Nettoyage léger en préservant le Markdown du LLM
    analysis_deanonymized = analysis_deanonymized.replace("\r\n", "\n")
    analysis_deanonymized = _normalize_table_pipes(analysis_deanonymized)
    analysis_deanonymized = _normalize_actions_section(analysis_deanonymized)
    analysis_deanonymized = _ensure_table_spacing(analysis_deanonymized)
    # normalise les tableaux Markdown pour conserver l'alignement dans les exports
    analysis_deanonymized = _normalize_markdown_tables(analysis_deanonymized)
    # supprime les lignes parasites dans la table des participants
    analysis_deanonymized = _prune_participants_table(
        analysis_deanonymized,
        mapping,
        role_hints,
        speaker_tags=speaker_tags,
        speaker_labels=speaker_labels,
    )
    analysis_deanonymized = _normalize_markdown_tables(analysis_deanonymized)
    # convertit les puces unicode en puces Markdown standard
    analysis_deanonymized = re.sub(r"^[ \t]*[•∙]", "- ", analysis_deanonymized, flags=re.MULTILINE)

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
    docx_dir = outputs_root / "docx"  # docx exports
    pdf_dir = outputs_root / "pdf"
    reports_dir.mkdir(parents=True, exist_ok=True)
    docx_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    out_txt_anon = reports_dir / f"{base}_meeting_report_anonymized.txt"
    out_txt = reports_dir / f"{base}_meeting_report.txt"
    out_md = reports_dir / f"{base}_meeting_report.md"
    out_docx = docx_dir / f"{base}_meeting_report.docx"
    out_pdf = pdf_dir / f"{base}_meeting_report.pdf"
    report_title = base.replace("_", " ").strip() or base

    out_txt_anon.write_text(analysis_anonymized, encoding="utf-8")
    out_txt.write_text(analysis_deanonymized, encoding="utf-8")
    out_md.write_text(analysis_deanonymized, encoding="utf-8")

    styled_markdown = analysis_deanonymized
    _convert_markdown_with_pandoc(styled_markdown, to="docx", out_path=out_docx)
    _render_pdf_report(styled_markdown, out_pdf, title=report_title)

    return {
        "report_anonymized_txt": str(out_txt_anon),
        "report_txt": str(out_txt),
        "report_markdown": str(out_md),
        "report_docx": str(out_docx),
        "report_pdf": str(out_pdf),
        "report_status": "generated",
        "report_reason": "",
    }
