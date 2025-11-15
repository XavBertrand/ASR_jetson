from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
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

_TAG_NORM_RE = re.compile(r"<\s*([a-zA-Z]+)\s*_(\s*\d+)\s*[^>]*>|\{\s*([a-zA-Z]+)\s*_(\s*\d+)\s*[^}]*\}", re.UNICODE)
_PERSON_TAG_RE = re.compile(r"<\s*PERSON\s*_(\s*\d+)\s*>", re.IGNORECASE)
_PANDOC_MD_FORMAT = "markdown+pipe_tables+grid_tables+multiline_tables+table_captions+raw_html"
_PREFERRED_SANS_FONTS: Tuple[str, ...] = ("Arial", "Calibri", "Liberation Sans", "DejaVu Sans")
_ROLE_KEYWORD_HINTS = {
    "delphine": "Delphine (avocat gérante)",
    "marie": "Delphine (avocat gérante)",
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


def _count_person_tags(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for match in _PERSON_TAG_RE.finditer(text):
        key = _normalize_tag_key(match.group(0))
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


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
    if "delphine" in lowered:
        return "Delphine (avocat gérante)"
    if "collaborateur" in lowered or "collaboratrice" in lowered:
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
                        if normalized_role in {"Delphine (avocat gérante)", "Collaborateur", "Client"}:
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
            if "delphine" in tail and "avocat" in tail:
                roles[tag] = "Delphine (avocat gérante)"
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


def _format_table_block(block: List[str]) -> List[str]:
    rows: List[List[str]] = []
    max_cols = 0

    for idx, raw_line in enumerate(block):
        trimmed = raw_line.strip().strip("|")
        if not trimmed:
            continue
        cells = [cell.strip() for cell in trimmed.split("|")]
        if idx == 0:
            while cells and all(ch in "-–—" for ch in cells[-1].replace(" ", "")):
                cells.pop()
        else:
            if all(
                not cell or all(ch in "-–—" for ch in cell.replace(" ", ""))
                for cell in cells
            ):
                continue
        if len(cells) <= 1:
            return block
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
        line = lines[i]
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
                current = lines[j]
                current_stripped = current.strip()
                if not current_stripped:
                    break
                if "|" in current:
                    block.append(current)
                    last_had_pipe = True
                    j += 1
                    continue
                if last_had_pipe and current.startswith(" "):
                    block[-1] = block[-1].rstrip() + " " + current_stripped
                    j += 1
                    continue
                break
            if len(block) >= 2:
                normalized_lines.extend(_format_table_block(block))
                i = j
                continue
        normalized_lines.append(line)
        i += 1

    return "\n".join(normalized_lines)


def _prune_participants_table(
    text: str,
    mapping: Dict[str, Any],
    roles: Dict[str, str],
) -> str:
    if not roles:
        return text

    lookup = _build_tag_lookup(mapping)
    allowed_keywords: Set[str] = set()
    for tag_key, role in roles.items():
        canonical = lookup.get(tag_key)
        if not canonical:
            continue
        canonical_lower = canonical.lower()
        if "delphine" in role.lower():
            allowed_keywords.add("delphine")
        else:
            allowed_keywords.add(canonical_lower)
    if not allowed_keywords:
        return text

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
                if not current.strip():
                    break
                if current.strip().startswith("### "):
                    break
                table_block.append(current)
                i += 1

            filtered_block: List[str] = []
            for block_line in table_block:
                stripped = block_line.strip()
                if not stripped.startswith("|"):
                    filtered_block.append(block_line)
                    continue
                cells = [cell.strip().lower() for cell in stripped.strip("|").split("|")]
                if all(not cell for cell in cells):
                    continue
                if all(ch == "-" for cell in cells for ch in cell.replace(" ", "")):
                    filtered_block.append(block_line)
                    continue
                row_text = " ".join(cells)
                if any(keyword in row_text for keyword in allowed_keywords):
                    filtered_block.append(block_line)
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


def _drop_unknown_tags(text: str, known_tags: Set[str]) -> str:
    if not known_tags:
        return text

    def _repl(match: re.Match[str]) -> str:
        tag = _normalize_tag_key(match.group(1))
        return "" if tag not in known_tags else match.group(0)

    return _INVENTED_TAG_RE.sub(_repl, text)


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
    format_spec = _PANDOC_MD_FORMAT

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
    4) Exporte TXT/Markdown et génère DOCX+PDF via pypandoc
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
    relevant_tags = _select_relevant_person_tags(mapping_raw, person_counts)

    # 2) Mistral Large
    user_payload = prompt.user_prefix + anon_text
    analysis_anonymized = mistral_client.chat_complete(model=prompt.model, system=prompt.system, user_text=user_payload)

    analysis_anonymized = _normalize_llm_placeholders(analysis_anonymized)
    role_hints = _extract_role_hints(analysis_anonymized, allowed_tags=relevant_tags)

    # 3) Désanonymisation
    mapping = _ensure_legacy_mapping(mapping_raw)
    role_hints = _refine_role_hints(role_hints, mapping, person_counts)
    analysis_deanonymized = deanonymize_text(analysis_anonymized, mapping, restore="canonical")
    analysis_deanonymized = _rewrite_roles(analysis_deanonymized, mapping, role_hints)
    analysis_deanonymized = _drop_unknown_tags(analysis_deanonymized, _collect_known_tags(mapping))

    # — Nettoyage léger —
    # normalise quelques séparateurs Markdown et petites incohérences de formatage
    analysis_deanonymized = analysis_deanonymized.replace("\r\n", "\n")
    analysis_deanonymized = re.sub(r"\n?---\n?", "\n\n", analysis_deanonymized)
    # tables à double "||" -> un seul "|"
    analysis_deanonymized = analysis_deanonymized.replace("||", "|")
    # s'assure que les puces commencent sur une nouvelle ligne
    analysis_deanonymized = re.sub(r"(\S)(-\s+\*\*|-\s+)", r"\1\n\2", analysis_deanonymized)
    # normalise les tableaux Markdown pour conserver l'alignement dans les exports
    analysis_deanonymized = _normalize_markdown_tables(analysis_deanonymized)
    # supprime les lignes parasites dans la table des participants
    analysis_deanonymized = _prune_participants_table(analysis_deanonymized, mapping, role_hints)
    analysis_deanonymized = _normalize_markdown_tables(analysis_deanonymized)
    # convertit les puces unicode en puces Markdown standard
    analysis_deanonymized = re.sub(r"^[ \t]*[•∙]", "- ", analysis_deanonymized, flags=re.MULTILINE)
    # garantit que toutes les entités canoniques apparaissent au moins une fois
    analysis_deanonymized = _append_missing_entities_section(analysis_deanonymized, mapping)

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

    out_txt_anon.write_text(analysis_anonymized, encoding="utf-8")
    out_txt.write_text(analysis_deanonymized, encoding="utf-8")
    out_md.write_text(analysis_deanonymized, encoding="utf-8")
    _convert_markdown_with_pandoc(analysis_deanonymized, to="docx", out_path=out_docx)
    _convert_markdown_with_pandoc(analysis_deanonymized, to="pdf", out_path=out_pdf)

    return {
        "report_anonymized_txt": str(out_txt_anon),
        "report_txt": str(out_txt),
        "report_markdown": str(out_md),
        "report_docx": str(out_docx),
        "report_pdf": str(out_pdf),
        "report_status": "generated",
        "report_reason": "",
    }
