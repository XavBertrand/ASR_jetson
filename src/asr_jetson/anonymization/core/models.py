"""Typed internal models for document anonymization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


FormatType = Literal["pdf", "docx", "xlsx", "txt"]
DocStatus = Literal["queued", "processing", "succeeded", "degraded", "failed"]
BatchStatus = Literal["completed", "completed_with_errors", "failed"]


@dataclass
class Policy:
    policy_id: str
    allow_network: bool = False
    enable_ner: bool = True
    enable_regex: bool = True
    enable_rules: bool = True
    continue_on_error: bool = True
    emit_mapping: bool = False
    mapping_required: bool = False
    max_documents_per_batch: int = 500
    max_pages_per_document: int = 1000
    max_total_input_mb: int = 1024
    mapping_schema_version: str = "1.0"
    report_schema_version: str = "1.0"
    storage_permissions_dir: int = 0o700
    storage_permissions_file: int = 0o600


@dataclass
class Span:
    span_id: str
    document_id: str
    start: int
    end: int
    anchor_type: str
    anchor_ref: str


@dataclass
class Entity:
    entity_id: str
    document_id: str
    span_id: str
    entity_type: str
    value: str
    detector_source: str
    confidence: float
    placeholder: str = ""


@dataclass
class ParsedDocument:
    document_id: str
    format: FormatType
    text: str
    spans: list[Span] = field(default_factory=list)


@dataclass
class DocumentRequest:
    case_id: str
    policy: Policy
    input_path: Path
    output_path: Path
    format_hint: FormatType | None = None


@dataclass
class DocumentResult:
    document_id: str
    status: DocStatus
    output_path: str | None
    warning_codes: list[str] = field(default_factory=list)
    failure_code: str | None = None
    failure_message_safe: str | None = None
    mapping_path: str | None = None


@dataclass
class BatchRequest:
    case_id: str
    policy_name: str
    policy: Policy
    input_paths: list[Path]
    output_root: Path
    report_path: Path
    mapping_mode: Literal["auto", "always", "never"] = "auto"
    continue_on_error: bool = True


@dataclass
class BatchResult:
    case_id: str
    status: BatchStatus
    totals: dict[str, int]
    documents: list[DocumentResult]
    report_path: str


@dataclass
class AuditEvent:
    event_code: str
    case_id: str
    document_id: str | None
    trace_id: str
    timestamp: str = field(default_factory=now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
