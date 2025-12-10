from __future__ import annotations

import json
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

_PANDOC_MD_FORMAT = (
    "markdown+pipe_tables+grid_tables+multiline_tables+table_captions+raw_html+fenced_divs"
    "-yaml_metadata_block"
)
_REPORT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "config" / "meeting.html"
_REPORT_CSS_PATH = Path(__file__).resolve().parent.parent / "config" / "report.css"


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


def _build_html_report(markdown_text: str, *, title: Optional[str] = None) -> str:
    if not _HAS_PYPANDOC:
        raise RuntimeError(
            "pypandoc is required to render HTML reports"
        ) from _PYPANDOC_IMPORT_ERROR

    if not _REPORT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Meeting report template missing: {_REPORT_TEMPLATE_PATH}")

    extra_args = ["--standalone", "--template", str(_REPORT_TEMPLATE_PATH)]
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

    if not _REPORT_CSS_PATH.exists():
        raise FileNotFoundError(f"Meeting report CSS missing: {_REPORT_CSS_PATH}")

    html_report = _build_html_report(markdown_text, title=title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_report, base_url=str(_REPORT_TEMPLATE_PATH.parent)).write_pdf(
        target=str(out_path),
        stylesheets=[CSS(filename=str(_REPORT_CSS_PATH))],
    )


def generate_pdf_report(
    anonymized_markdown_path: Path,
    mapping_json_path: Path,
    output_dir: Path,
    *,
    run_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, str]:
    """
    Produce a deanonymized Markdown report and render it to PDF using the shared HTML/CSS assets.
    """
    anonymized_md = _load_markdown(Path(anonymized_markdown_path))
    mapping = _normalize_mapping(_load_mapping(Path(mapping_json_path)))
    deanonymized_md = deanonymize_report_markdown(anonymized_md, mapping)

    base = _derive_base_name(Path(anonymized_markdown_path), run_id=run_id)
    reports_dir = Path(output_dir) / "reports"
    pdf_dir = Path(output_dir) / "pdf"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    md_path = reports_dir / f"{base}_meeting_report.md"
    pdf_path = pdf_dir / f"{base}_meeting_report.pdf"
    md_path.write_text(deanonymized_md, encoding="utf-8")

    report_title = title or base.replace("_", " ").strip() or base
    _render_pdf_report(deanonymized_md, pdf_path, title=report_title)

    return {
        "report_markdown": str(md_path),
        "report_pdf": str(pdf_path),
        "report_status": "generated",
        "report_reason": "",
    }
