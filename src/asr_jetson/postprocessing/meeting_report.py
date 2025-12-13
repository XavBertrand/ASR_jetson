from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from asr_jetson.postprocessing.anonymizer import deanonymize_text

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

try:  # pragma: no cover - optional dependency
    import language_tool_python  # type: ignore
    from language_tool_python import utils as lt_utils  # type: ignore

    _HAS_LANGUAGETOOL = True
    _LANGUAGETOOL_IMPORT_ERROR: Optional[Exception] = None
except Exception as _err:  # pragma: no cover - executed when language_tool_python missing
    language_tool_python = None  # type: ignore
    lt_utils = None  # type: ignore
    _HAS_LANGUAGETOOL = False
    _LANGUAGETOOL_IMPORT_ERROR = _err

_PANDOC_MD_FORMAT = (
    "markdown+pipe_tables+grid_tables+multiline_tables+table_captions+raw_html+fenced_divs"
    "-yaml_metadata_block"
)
_REPORT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "config" / "meeting.html"
_REPORT_CSS_PATH = Path(__file__).resolve().parent.parent / "config" / "report.css"
DEFAULT_REPORT_TITLE = "Compte Rendu d'Entretien Collaborateur"
PROMPT_TITLE_MAP: dict[str, str] = {
    "entretien_collaborateur": "Compte Rendu d'Entretien Collaborateur",
    "entretien_client_particulier_contentieux": "Compte Rendu d'Entretien Client",
    "entretien_client_professionnel_conseil": "Compte Rendu d'Entretien Client",
    "entretien_client_professionnel_contentieux": "Compte Rendu d'Entretien Client",
}
_LT_ENDPOINT = os.getenv("LT_ENDPOINT", "").strip() or None
_LT_DISABLED = (os.getenv("DISABLE_LANGUAGETOOL") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_LT_INIT_DONE = False
_LT_INIT_ERROR: Optional[str] = None
_LT_TOOL: Optional[Any] = None


def _prepare_lt_home() -> Optional[str]:
    """
    Ensure a writable LanguageTool cache directory exists and is exported.
    """
    env_value = os.environ.get("LT_HOME")
    candidates = []
    if env_value:
        candidates.append(Path(env_value))
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            candidates.append(Path(xdg_cache) / "LanguageTool")
        candidates.append(Path.home() / ".cache" / "LanguageTool")
        candidates.append(Path.cwd() / ".cache" / "LanguageTool")
        candidates.append(Path(tempfile.gettempdir()) / "LanguageTool")

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        resolved = str(path)
        os.environ.setdefault("LT_HOME", resolved)
        os.environ.setdefault("LTP_PATH", resolved)
        return resolved
    return None


def _ensure_language_tool() -> Optional[Any]:
    """
    Lazily instantiate LanguageTool (French) with optional remote endpoint support.
    """
    global _LT_INIT_DONE, _LT_INIT_ERROR, _LT_TOOL
    if _LT_TOOL is not None:
        return _LT_TOOL
    if _LT_DISABLED or _LT_INIT_DONE or not _HAS_LANGUAGETOOL:
        _LT_INIT_DONE = True
        return None

    _LT_INIT_DONE = True
    try:
        cache_dir = _prepare_lt_home()
        if cache_dir:
            os.environ.setdefault("LTP_PATH", cache_dir)

        tool_cls = language_tool_python.LanguageTool  # type: ignore[attr-defined]
        _LT_TOOL = tool_cls("fr", remote_server=_LT_ENDPOINT) if _LT_ENDPOINT else tool_cls("fr")
        return _LT_TOOL
    except Exception as err:
        _LT_INIT_ERROR = str(err)
        _LT_TOOL = None
        print(f"⚠️ LanguageTool unavailable: {err}")
        return None


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
                    "mentions": mentions,
                }
            )
        normalized = dict(mapping)
        normalized["entities"] = normalized_entities
        return normalized
    return mapping


def deanonymize_report_markdown(anonymized_markdown: str, mapping: Dict[str, Any]) -> str:
    """
    Replace pseudonyms with their canonical values using the anonymization mapping.
    """
    return deanonymize_text(anonymized_markdown, mapping, restore="canonical")


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

    if not _REPORT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Meeting report template missing: {_REPORT_TEMPLATE_PATH}")

    extra_args = ["--standalone", "--template", str(_REPORT_TEMPLATE_PATH)]
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

    if not _REPORT_CSS_PATH.exists():
        raise FileNotFoundError(f"Meeting report CSS missing: {_REPORT_CSS_PATH}")

    html_report = _build_html_report(markdown_text, title=title, report_date=report_date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_report, base_url=str(_REPORT_TEMPLATE_PATH.parent)).write_pdf(
        target=str(out_path),
        stylesheets=[CSS(filename=str(_REPORT_CSS_PATH))],
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
    meeting_date_str = (
        (meeting_date or datetime.now().strftime("%Y-%m-%d")).strip()
        or datetime.now().strftime("%Y-%m-%d")
    )
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
        date=_safe_filename_component(meeting_date_str, "date"),
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
        report_date=meeting_date_str,
    )
    _render_docx_report(
        corrected_md,
        docx_path,
        title=report_title,
        report_date=meeting_date_str,
    )

    return {
        "report_markdown": str(md_path),
        "report_pdf": str(pdf_path),
        "report_docx": str(docx_path),
        "report_status": "generated",
        "report_reason": "",
    }
