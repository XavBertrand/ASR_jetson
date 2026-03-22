from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from asr_jetson.postprocessing.anonymizer import deanonymize_text
from asr_jetson.postprocessing.languagetool_helper import ensure_language_tool

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

_PANDOC_MD_FORMAT = (
    "markdown+pipe_tables+grid_tables+multiline_tables+table_captions+raw_html+fenced_divs"
    "-yaml_metadata_block"
)
def _resolve_config_dir() -> Path:
    default_dir = Path(__file__).resolve().parents[1] / "config"
    env_value = os.environ.get("ASR_CONFIG_DIR")
    if not env_value:
        return default_dir
    env_path = Path(env_value).expanduser()
    if not env_path.is_absolute():
        env_path = (Path.cwd() / env_path).resolve()
    return env_path


def _report_template_path() -> Path:
    return _resolve_config_dir() / "meeting.html"


def _report_css_path() -> Path:
    return _resolve_config_dir() / "report.css"
DEFAULT_REPORT_TITLE = "Compte Rendu d'Entretien Collaborateur"
PROMPT_TITLE_MAP: dict[str, str] = {
    "entretien_collaborateur": "Compte Rendu d'Entretien Collaborateur",
    "entretien_client_particulier_contentieux": "Compte Rendu d'Entretien Client",
    "entretien_client_professionnel_conseil": "Compte Rendu d'Entretien Client",
    "entretien_client_professionnel_contentieux": "Compte Rendu d'Entretien Client",
    "compte_rendu_association": "Compte Rendu Association",
}
def _ensure_language_tool() -> Optional[Any]:
    """
    Lazily instantiate LanguageTool (French) with optional remote endpoint support.
    """
    return ensure_language_tool()


def _polish_markdown_with_languagetool(markdown_text: str) -> str:
    """
    Apply LanguageTool corrections to the deanonymized report Markdown, restricted to
    punctuation/spacing and casing changes to avoid altering names.
    """
    tool = _ensure_language_tool()
    if tool is None:
        return markdown_text
    try:
        matches = tool.check(markdown_text)

        def _tokens(text: str) -> list[str]:
            # Keep alphanumeric tokens (including accents) to preserve names.
            return re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", text)

        def _is_safe_replacement(src: str, repl: str) -> bool:
            # Allow only punctuation/spacing/casing changes; forbid lexical edits/splits.
            src_tokens = _tokens(src)
            repl_tokens = _tokens(repl)
            if len(src_tokens) != len(repl_tokens):
                return False
            if [t.casefold() for t in src_tokens] != [t.casefold() for t in repl_tokens]:
                return False
            return True

        edits = []
        for mt in matches:
            repl = mt.replacements[0] if mt.replacements else None
            if not repl:
                continue
            src_slice = markdown_text[mt.offset : mt.offset + mt.errorLength]
            if _is_safe_replacement(src_slice, repl):
                edits.append((mt.offset, mt.errorLength, repl))

        if not edits:
            return markdown_text

        edits.sort(key=lambda x: x[0], reverse=True)
        buf = markdown_text
        for off, ln, repl in edits:
            buf = buf[:off] + repl + buf[off + ln :]
        return buf
    except Exception as err:
        print(f"⚠️ LanguageTool correction skipped: {err}")
        return markdown_text


def pdf_export_prerequisites() -> list[str]:
    """
    Return a list of missing optional dependencies required to export PDFs.
    """
    missing: list[str] = []
    if not _HAS_PYPANDOC:
        missing.append("pypandoc")
    if not _HAS_WEASYPRINT:
        missing.append("weasyprint")
    return missing


def _load_markdown(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Anonymized report not found: {path}")
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def _load_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Harmonize mapping structures so ``deanonymize_text`` can consume them.
    Accept both dict-based and list-based ``entities``.
    """
    entities = mapping.get("entities")
    if isinstance(entities, dict):
        normalized_entities = []
        for tag, info in entities.items():
            canonical = info.get("canonical") or (info.get("values") or [None])[0] or tag
            mentions = info.get("values") or info.get("variants") or []
            normalized_entities.append(
                {
                    "tag": tag,
                    "type": info.get("label") or info.get("type"),
                    "canonical": canonical,
                    "pseudonym": info.get("pseudonym"),
                    "mentions": mentions,
                }
            )
        normalized = dict(mapping)
        normalized["entities"] = normalized_entities
        return normalized
    return mapping


_NAME_SEQUENCE_RE = re.compile(
    r"\b[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)?"
    r"(?:\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)?){1,2}\b"
)
_COMMON_REPORT_WORDS = {
    "compte", "rendu", "entretien", "client", "collaborateur", "association",
    "résumé", "resume", "participants", "sujets", "décisions", "decisions",
    "actions", "prochaines", "étapes", "etapes", "analyse", "risques",
    "opportunités", "opportunites", "chiffres", "repères", "reperes",
    "points", "friction", "difficultés", "difficultes", "contexte",
    "description", "situation", "thématiques", "thematiques", "stratégie",
    "strategie", "plan", "vigilance", "non", "précisé", "precise", "precisé",
}
_COMMON_MONTHS = {
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin", "juillet",
    "août", "aout", "septembre", "octobre", "novembre", "décembre", "decembre",
}
_COMMON_DAYS = {
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
}
_FRENCH_MONTH_LABELS = {
    1: "janvier",
    2: "février",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "août",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "décembre",
}
_DATE_PARSE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%d/%m/%Y", "%d-%m-%Y", "%Y%m%d")
_PERSON_NAME_CLEAN_RE = re.compile(r"[^a-zà-öø-ÿ\s]+")
_PERSON_NAME_HYPHENS_RE = re.compile(r"[’'`´\-\u2010\u2011\u2012\u2013\u2014\u2212]+")
_PERSON_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+$")
_PERSON_ALIAS_BOUNDARY_RE = r"[A-Za-zÀ-ÖØ-öø-ÿ'’\-]"
_PERSON_TITLE_PREFIXES = {
    "m",
    "m.",
    "mme",
    "mme.",
    "mlle",
    "mlle.",
    "monsieur",
    "madame",
    "mademoiselle",
    "maitre",
    "maître",
    "me",
    "dr",
    "docteur",
    "docteure",
}
_ORG_NAME_SUFFIXES = {
    "conseil",
    "legal",
    "services",
    "industries",
    "groupe",
    "solutions",
    "partners",
    "associates",
    "collectif",
    "studio",
}


def _build_allowed_person_names(mapping: Dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    entities = mapping.get("entities") or []
    if isinstance(entities, dict):
        items = entities.values()
    else:
        items = entities
    for info in items:
        label = (info.get("label") or info.get("type") or "").upper()
        if label not in {"PERSON", "PER"}:
            continue
        for key in ("canonical", "pseudonym"):
            value = info.get(key)
            if value:
                allowed.add(value.lower())
        for key in ("values", "variants", "mentions"):
            values = info.get(key) or []
            for value in values:
                if value:
                    allowed.add(str(value).lower())
    pseudo_reverse = mapping.get("pseudonym_reverse_map") or {}
    if isinstance(pseudo_reverse, dict):
        for pseudo, canonical in pseudo_reverse.items():
            if pseudo:
                allowed.add(str(pseudo).lower())
            if canonical:
                allowed.add(str(canonical).lower())
    context_names = mapping.get("context_names") or []
    if isinstance(context_names, list):
        for name in context_names:
            if name:
                allowed.add(str(name).lower())
    return allowed


def _normalize_person_name(value: str) -> str:
    lowered = value.lower()
    lowered = _PERSON_NAME_HYPHENS_RE.sub(" ", lowered)
    lowered = _PERSON_NAME_CLEAN_RE.sub(" ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _build_context_name_aliases(mapping: Dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    context_names = mapping.get("context_names") or []
    if not isinstance(context_names, list):
        return aliases
    for name in context_names:
        if not name:
            continue
        display = str(name).strip()
        normalized = _normalize_person_name(display)
        if normalized and normalized not in aliases:
            aliases[normalized] = display
    return aliases


def _person_name_tokens(value: str) -> list[str]:
    tokens = [token.strip(" ,.;:()[]{}\"") for token in re.split(r"\s+", value.strip()) if token]
    while tokens:
        first = tokens[0].lower().strip(".,;:")
        if first in _PERSON_TITLE_PREFIXES:
            tokens.pop(0)
        else:
            break
    return [token for token in tokens if token]


def _looks_like_person_pseudonym(value: str) -> bool:
    tokens = _person_name_tokens(value)
    if len(tokens) < 2:
        return False
    if tokens[-1].lower().strip(".,;:") in _ORG_NAME_SUFFIXES:
        return False
    for token in tokens:
        if not _PERSON_TOKEN_RE.fullmatch(token):
            return False
        if not token[0].isupper():
            return False
    return True


def _iter_person_pseudonym_pairs(mapping: Dict[str, Any]) -> list[tuple[str, str]]:
    pseudo_reverse = mapping.get("pseudonym_reverse_map") or {}
    if not isinstance(pseudo_reverse, dict):
        return []

    person_pseudonyms: set[str] = set()
    entities = mapping.get("entities") or []
    pseudonym_map = mapping.get("pseudonym_map") or {}
    entity_items: list[tuple[str, Dict[str, Any]]] = []

    if isinstance(entities, dict):
        for tag, info in entities.items():
            if isinstance(info, dict):
                entity_items.append((str(tag), info))
    elif isinstance(entities, list):
        for info in entities:
            if isinstance(info, dict):
                entity_items.append((str(info.get("tag") or ""), info))

    for tag, info in entity_items:
        label = (info.get("label") or info.get("type") or "").upper()
        if label not in {"PERSON", "PER"}:
            continue
        pseudo = info.get("pseudonym")
        if not pseudo and isinstance(pseudonym_map, dict) and tag:
            pseudo = pseudonym_map.get(tag)
        if pseudo:
            person_pseudonyms.add(str(pseudo))

    pairs: list[tuple[str, str]] = []
    if person_pseudonyms:
        for pseudo in person_pseudonyms:
            canonical = pseudo_reverse.get(pseudo)
            if pseudo and canonical:
                pairs.append((str(pseudo), str(canonical)))
        return pairs

    for pseudo, canonical in pseudo_reverse.items():
        pseudo_value = str(pseudo)
        canonical_value = str(canonical)
        if pseudo_value and canonical_value and _looks_like_person_pseudonym(pseudo_value):
            pairs.append((pseudo_value, canonical_value))
    return pairs


def _build_person_first_name_aliases(mapping: Dict[str, Any]) -> dict[str, str]:
    alias_candidates: dict[str, set[str]] = {}
    alias_display: dict[str, str] = {}

    for pseudonym, canonical in _iter_person_pseudonym_pairs(mapping):
        tokens = _person_name_tokens(pseudonym)
        if len(tokens) < 2:
            continue
        first_name = tokens[0]
        if not first_name or first_name == canonical:
            continue
        normalized = _normalize_person_name(first_name)
        if not normalized:
            continue
        if normalized in _COMMON_REPORT_WORDS or normalized in _COMMON_MONTHS or normalized in _COMMON_DAYS:
            continue
        alias_display.setdefault(normalized, first_name)
        alias_candidates.setdefault(normalized, set()).add(canonical)

    aliases: dict[str, str] = {}
    for normalized, canonicals in alias_candidates.items():
        if len(canonicals) == 1:
            alias = alias_display.get(normalized)
            canonical = next(iter(canonicals))
            if alias and canonical and alias != canonical:
                aliases[alias] = canonical
    return aliases


def _restore_person_first_name_aliases(text: str, mapping: Dict[str, Any]) -> str:
    aliases = _build_person_first_name_aliases(mapping)
    if not aliases:
        return text
    restored = text
    for alias, canonical in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(
            rf"(?<!{_PERSON_ALIAS_BOUNDARY_RE}){re.escape(alias)}(?!{_PERSON_ALIAS_BOUNDARY_RE})"
        )
        restored = pattern.sub(canonical, restored)
    return restored


def _strip_unknown_person_names(text: str, mapping: Dict[str, Any]) -> str:
    allowed = _build_allowed_person_names(mapping)
    if not allowed:
        return text
    allowed_normalized = {_normalize_person_name(name) for name in allowed if name}
    context_aliases = _build_context_name_aliases(mapping)

    def _replace(match: re.Match) -> str:
        candidate = match.group(0)
        lower = candidate.lower()
        normalized = _normalize_person_name(candidate)
        if lower in allowed:
            return candidate
        if normalized and normalized in allowed_normalized:
            return candidate
        if lower in _COMMON_REPORT_WORDS or lower in _COMMON_MONTHS or lower in _COMMON_DAYS:
            return candidate
        if candidate.isupper():
            return candidate
        if normalized:
            for context_normalized, context_display in sorted(
                context_aliases.items(), key=lambda item: len(item[0]), reverse=True
            ):
                if normalized == context_normalized:
                    return context_display
                if normalized.startswith(f"{context_normalized} "):
                    return context_display
                if " " not in context_normalized and context_normalized in normalized.split():
                    return context_display
        return ""

    updated = _NAME_SEQUENCE_RE.sub(_replace, text)
    lines = updated.splitlines()
    cleaned = []
    for line in lines:
        match = re.match(r"^([ \t]*)(.*)$", line)
        if not match:
            cleaned.append(line)
            continue
        indent_text, rest = match.groups()
        rest = re.sub(r"[ \t]{2,}", " ", rest)
        rest = re.sub(r"[ \t]+([,.;:!?])", r"\1", rest)
        cleaned.append(indent_text + rest)
    return "\n".join(cleaned)


def deanonymize_report_markdown(anonymized_markdown: str, mapping: Dict[str, Any]) -> str:
    """
    Replace pseudonyms with their canonical values using the anonymization mapping.
    Also restores single first-name mentions when the PERSON mapping is unambiguous.
    """
    restored = deanonymize_text(anonymized_markdown, mapping, restore="canonical")
    restored = re.sub(r"\bd[’'](?=(?:M\.?|Mme\.?|Monsieur|Madame)\b)", "de ", restored)
    return restored


def _derive_base_name(anonymized_path: Path, run_id: Optional[str] = None) -> str:
    if run_id:
        return run_id
    stem = anonymized_path.stem
    for suffix in ("_anon_clean", "_anonymized", "_anon"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _safe_filename_component(component: str, fallback: str) -> str:
    """
    Sanitize a string to be file-system friendly while keeping it readable.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", component.strip())
    return cleaned or fallback


def format_meeting_date_literal(date_value: Optional[str]) -> str:
    """
    Format the meeting date in a French literal form (e.g., "15 janvier 2012").
    """
    raw = (date_value or "").strip()
    if not raw:
        now = datetime.now()
        return f"{now.day} {_FRENCH_MONTH_LABELS[now.month]} {now.year}"

    lower = raw.lower()
    if any(month in lower for month in _COMMON_MONTHS):
        return raw

    candidate = raw.split("T", 1)[0].split(" ", 1)[0].strip(" ,;")
    for fmt in _DATE_PARSE_FORMATS:
        try:
            parsed = datetime.strptime(candidate, fmt)
        except ValueError:
            continue
        return f"{parsed.day} {_FRENCH_MONTH_LABELS[parsed.month]} {parsed.year}"

    return raw


def _build_html_report(
    markdown_text: str,
    *,
    title: Optional[str] = None,
    report_date: Optional[str] = None,
) -> str:
    if not _HAS_PYPANDOC:
        raise RuntimeError(
            "pypandoc is required to render HTML reports"
        ) from _PYPANDOC_IMPORT_ERROR

    template_path = _report_template_path()
    if not template_path.exists():
        raise FileNotFoundError(f"Meeting report template missing: {template_path}")

    extra_args = ["--standalone", "--template", str(template_path)]
    date_value = report_date or datetime.now().strftime("%d/%m/%Y")
    extra_args.extend(["--metadata", f"date={date_value}"])
    if title:
        extra_args.extend(["--metadata", f"title={title}"])

    return pypandoc.convert_text(  # type: ignore[call-arg]
        markdown_text,
        to="html",
        format=_PANDOC_MD_FORMAT,
        extra_args=extra_args,
    )


def _render_pdf_report(
    markdown_text: str,
    out_path: Path,
    *,
    title: Optional[str] = None,
    report_date: Optional[str] = None,
) -> None:
    if not _HAS_WEASYPRINT:
        raise RuntimeError(
            "weasyprint is required to export meeting reports to PDF"
        ) from _WEASYPRINT_IMPORT_ERROR

    template_path = _report_template_path()
    css_path = _report_css_path()
    if not css_path.exists():
        raise FileNotFoundError(f"Meeting report CSS missing: {css_path}")

    html_report = _build_html_report(markdown_text, title=title, report_date=report_date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_report, base_url=str(template_path.parent)).write_pdf(
        target=str(out_path),
        stylesheets=[CSS(filename=str(css_path))],
    )


def _render_docx_report(
    markdown_text: str,
    out_path: Path,
    *,
    title: Optional[str] = None,
    report_date: Optional[str] = None,
) -> None:
    if not _HAS_PYPANDOC:
        raise RuntimeError(
            "pypandoc is required to export meeting reports to DOCX"
        ) from _PYPANDOC_IMPORT_ERROR

    extra_args = []
    date_value = report_date or datetime.now().strftime("%d/%m/%Y")
    extra_args.extend(["--metadata", f"date={date_value}"])
    if title:
        extra_args.extend(["--metadata", f"title={title}"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pypandoc.convert_text(  # type: ignore[call-arg]
        markdown_text,
        to="docx",
        format=_PANDOC_MD_FORMAT,
        outputfile=str(out_path),
        extra_args=extra_args,
    )


def resolve_default_report_title(prompt_key: Optional[str]) -> str:
    """
    Return a default report title based on the selected prompt category.
    """
    if prompt_key:
        return PROMPT_TITLE_MAP.get(prompt_key, DEFAULT_REPORT_TITLE)
    return DEFAULT_REPORT_TITLE


def generate_pdf_report(
    anonymized_markdown_path: Path,
    mapping_json_path: Path,
    output_dir: Path,
    *,
    run_id: Optional[str] = None,
    title: Optional[str] = None,
    prompt_key: Optional[str] = None,
    meeting_date: Optional[str] = None,
    audio_stem: Optional[str] = None,
    run_time: Optional[str] = None,
) -> Dict[str, str]:
    """
    Produce a deanonymized Markdown report and render it to PDF using the shared HTML/CSS assets.
    Defaults the rendered title based on the prompt category (collaborateur vs client)
    when none is provided.
    """
    anonymized_md = _load_markdown(Path(anonymized_markdown_path))
    mapping = _normalize_mapping(_load_mapping(Path(mapping_json_path)))
    deanonymized_md = deanonymize_report_markdown(anonymized_md, mapping)
    corrected_md = _polish_markdown_with_languagetool(deanonymized_md)

    base = _derive_base_name(Path(anonymized_markdown_path), run_id=run_id)
    meeting_date_raw = (
        (meeting_date or datetime.now().strftime("%Y-%m-%d")).strip()
        or datetime.now().strftime("%Y-%m-%d")
    )
    meeting_date_label = format_meeting_date_literal(meeting_date_raw)
    run_time_str = (
        (run_time or datetime.now().strftime("%H%M%S")).strip()
        or datetime.now().strftime("%H%M%S")
    )
    audio_component = str(audio_stem or base)
    reports_dir = Path(output_dir) / "reports"
    pdf_dir = Path(output_dir) / "pdf"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    md_path = reports_dir / f"{base}_meeting_report.md"
    pdf_filename = "compte_rendu_{audio}_{date}_{time}.pdf".format(
        audio=_safe_filename_component(audio_component, "audio"),
        date=_safe_filename_component(meeting_date_raw, "date"),
        time=_safe_filename_component(run_time_str, "time"),
    )
    pdf_path = pdf_dir / pdf_filename
    docx_path = pdf_path.with_suffix(".docx")
    md_path.write_text(corrected_md, encoding="utf-8")

    report_title = title or resolve_default_report_title(prompt_key)
    _render_pdf_report(
        corrected_md,
        pdf_path,
        title=report_title,
        report_date=meeting_date_label,
    )
    _render_docx_report(
        corrected_md,
        docx_path,
        title=report_title,
        report_date=meeting_date_label,
    )

    return {
        "report_markdown": str(md_path),
        "report_pdf": str(pdf_path),
        "report_docx": str(docx_path),
        "report_status": "generated",
        "report_reason": "",
    }
