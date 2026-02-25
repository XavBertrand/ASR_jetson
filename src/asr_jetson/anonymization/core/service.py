"""Document anonymization orchestration service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from asr_jetson.anonymization.core.errors import InputValidationError, ProcessingError
from asr_jetson.anonymization.core.models import (
    BatchRequest,
    BatchResult,
    DocumentRequest,
    DocumentResult,
    Entity,
    ParsedDocument,
)
from asr_jetson.anonymization.core.placeholders import PlaceholderGenerator, normalize_entity_value
from asr_jetson.anonymization.core.safe_logging import get_logger, sanitize_exception
from asr_jetson.anonymization.core.tempfiles import TempWorkspace
from asr_jetson.anonymization.detectors.ner_detector import NERUnavailableError, NerDetector
from asr_jetson.anonymization.detectors.regex_detector import RegexDetector
from asr_jetson.anonymization.detectors.rule_detector import RuleDetector
from asr_jetson.anonymization.parsers.docx_parser import DocxParser
from asr_jetson.anonymization.parsers.pdf_parser import PdfParser
from asr_jetson.anonymization.parsers.txt_parser import TxtParser
from asr_jetson.anonymization.parsers.xlsx_parser import XlsxParser
from asr_jetson.anonymization.renderers.docx_renderer import DocxRenderer
from asr_jetson.anonymization.renderers.pdf_renderer import PdfRenderer
from asr_jetson.anonymization.renderers.txt_renderer import TxtRenderer
from asr_jetson.anonymization.renderers.xlsx_renderer import XlsxRenderer
from asr_jetson.anonymization.storage.fs_security import ensure_dir_permissions, ensure_file_permissions
from asr_jetson.anonymization.storage.mapping_store import MappingStore


def _doc_id(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    return f"doc-{digest}"


def _ext(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def _build_replacements(entities: list[Entity], generator: PlaceholderGenerator) -> dict[str, str]:
    dedup: dict[tuple[str, str], str] = {}
    for entity in entities:
        value = entity.value.strip()
        if not value:
            continue
        key = (entity.entity_type.lower(), normalize_entity_value(value))
        dedup[key] = value

    replacements: dict[str, str] = {}
    for entity_type, normalized in sorted(dedup.keys()):
        original = dedup[(entity_type, normalized)]
        replacements[original] = generator.generate(entity_type, original)
    return replacements


class DocumentAnonymizer:
    """Batch-oriented anonymizer service with per-document isolation."""

    def __init__(self, mapping_store: MappingStore | None = None) -> None:
        self.logger = get_logger()
        self._parsers = {
            "txt": TxtParser(),
            "pdf": PdfParser(),
            "docx": DocxParser(),
            "xlsx": XlsxParser(),
        }
        self._renderers = {
            "txt": TxtRenderer(),
            "pdf": PdfRenderer(),
            "docx": DocxRenderer(),
            "xlsx": XlsxRenderer(),
        }
        self._regex = RegexDetector()
        self._ner = NerDetector()
        self._rules = RuleDetector()
        self._mapping_store = mapping_store or MappingStore()

    def _detect_entities(self, parsed: ParsedDocument, request: DocumentRequest, warning_codes: list[str]) -> list[Entity]:
        entities: list[Entity] = []
        if parsed.text.strip() == "":
            return entities

        if request.policy.enable_ner:
            try:
                entities.extend(self._ner.detect(parsed.text, parsed.document_id))
            except NERUnavailableError:
                warning_codes.append("NER_UNAVAILABLE")

        if request.policy.enable_regex:
            entities.extend(self._regex.detect(parsed.text, parsed.document_id))
        if request.policy.enable_rules:
            entities.extend(self._rules.detect(parsed.text, parsed.document_id))

        by_identity: dict[tuple[str, str, str], Entity] = {}
        for entity in entities:
            key = (entity.entity_type, normalize_entity_value(entity.value), entity.span_id)
            by_identity[key] = entity
        return list(by_identity.values())

    def _anonymize_document_with_generator(
        self,
        request: DocumentRequest,
        placeholder_generator: PlaceholderGenerator,
    ) -> tuple[DocumentResult, dict[str, str]]:
        input_path = request.input_path
        extension = request.format_hint or _ext(input_path)
        document_id = _doc_id(input_path)

        if extension not in self._parsers:
            raise InputValidationError(f"Unsupported format: {extension}")

        warning_codes: list[str] = []
        try:
            parsed = self._parsers[extension].parse(input_path, document_id)
            entities = self._detect_entities(parsed, request, warning_codes)
            replacements = _build_replacements(entities, placeholder_generator)
            self._renderers[extension].render(input_path, request.output_path, replacements)
            ensure_file_permissions(request.output_path, request.policy.storage_permissions_file)
            status = "degraded" if warning_codes else "succeeded"
            return (
                DocumentResult(
                    document_id=document_id,
                    status=status,
                    output_path=str(request.output_path),
                    warning_codes=warning_codes,
                ),
                replacements,
            )
        except Exception as exc:  # noqa: BLE001
            safe = sanitize_exception(exc)
            raise ProcessingError(safe) from exc

    def anonymize_document(self, request: DocumentRequest) -> DocumentResult:
        generator = PlaceholderGenerator(
            case_id=request.case_id,
            schema_version=request.policy.mapping_schema_version,
        )
        result, _ = self._anonymize_document_with_generator(request, generator)
        return result

    def _should_emit_mapping(self, request: BatchRequest) -> bool:
        mode = request.mapping_mode
        if mode == "always":
            return True
        if mode == "never":
            return False
        return bool(request.policy.emit_mapping)

    def anonymize_batch(self, request: BatchRequest) -> BatchResult:
        output_anonymized = request.output_root / "anonymized"
        ensure_dir_permissions(output_anonymized, request.policy.storage_permissions_dir)

        emit_mapping = self._should_emit_mapping(request)
        if request.policy.mapping_required and not emit_mapping:
            raise InputValidationError("Mapping emission required by policy")

        results: list[DocumentResult] = []
        failed = 0
        degraded = 0
        placeholder_generator = PlaceholderGenerator(
            case_id=request.case_id,
            schema_version=request.policy.mapping_schema_version,
        )

        with TempWorkspace(request.output_root):
            for input_path in request.input_paths:
                try:
                    out_path = output_anonymized / input_path.name
                    doc_request = DocumentRequest(
                        case_id=request.case_id,
                        policy=request.policy,
                        input_path=input_path,
                        output_path=out_path,
                        format_hint=_ext(input_path),
                    )
                    doc_result, replacements = self._anonymize_document_with_generator(
                        doc_request,
                        placeholder_generator,
                    )

                    if emit_mapping and doc_result.status != "failed":
                        mapping_path = self._mapping_store.write_mapping(
                            case_id=request.case_id,
                            document_id=doc_result.document_id,
                            schema_version=request.policy.mapping_schema_version,
                            mapping=replacements,
                            output_root=request.output_root,
                            file_mode=request.policy.storage_permissions_file,
                            dir_mode=request.policy.storage_permissions_dir,
                        )
                        doc_result.mapping_path = str(mapping_path)

                    results.append(doc_result)
                    if doc_result.status == "degraded":
                        degraded += 1
                except ProcessingError as exc:
                    failed += 1
                    results.append(
                        DocumentResult(
                            document_id=_doc_id(input_path),
                            status="failed",
                            output_path=None,
                            failure_code=exc.code,
                            failure_message_safe=exc.message_safe,
                        )
                    )
                    self.logger.warning("document_failed code=%s", exc.code)
                    if not request.continue_on_error:
                        break

        totals = {
            "total_documents": len(request.input_paths),
            "succeeded": len([r for r in results if r.status == "succeeded"]),
            "failed": failed,
            "degraded": degraded,
        }
        status = "completed" if failed == 0 else "completed_with_errors"
        report_payload = {
            "schema_version": request.policy.report_schema_version,
            "mapping_schema_version": request.policy.mapping_schema_version,
            "case_id": request.case_id,
            "policy": request.policy_name,
            "totals": totals,
            "documents": [r.__dict__ for r in results],
        }
        request.report_path.parent.mkdir(parents=True, exist_ok=True)
        request.report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        ensure_file_permissions(request.report_path, request.policy.storage_permissions_file)

        return BatchResult(
            case_id=request.case_id,
            status=status,
            totals=totals,
            documents=results,
            report_path=str(request.report_path),
        )
