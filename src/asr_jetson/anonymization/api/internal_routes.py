"""Internal API scaffolding for anonymization job lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asr_jetson.anonymization.core.errors import (
    AnonymizationError,
    InputValidationError,
    PolicyValidationError,
    SecurityPolicyError,
)
from asr_jetson.anonymization.core.models import BatchRequest, BatchResult
from asr_jetson.anonymization.core.policy import load_policy
from asr_jetson.anonymization.core.service import DocumentAnonymizer
from asr_jetson.anonymization.storage.mapping_store import MappingStore


@dataclass
class JobRecord:
    job_id: str
    status: str
    payload: dict[str, Any]
    case_id: str
    report_path: str
    network_mode: str
    totals: dict[str, int] = field(
        default_factory=lambda: {"total_documents": 0, "succeeded": 0, "failed": 0, "degraded": 0}
    )
    documents: list[dict[str, Any]] = field(default_factory=list)


class InternalJobStore:
    """In-memory API facade that mirrors contract behavior for tests/integration."""

    def __init__(self, service: DocumentAnonymizer | None = None) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency_index: dict[str, tuple[str, str]] = {}
        self._service = service or DocumentAnonymizer()
        self._mapping_store = MappingStore()

    @staticmethod
    def _payload_digest(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _error(status: int, code: str, message_safe: str) -> tuple[int, dict[str, Any]]:
        return status, {"code": code, "message_safe": message_safe}

    @staticmethod
    def _parse_input_paths(raw_inputs: Any) -> list[Path]:
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise InputValidationError("Invalid create job payload")

        paths: list[Path] = []
        for item in raw_inputs:
            if not isinstance(item, dict):
                raise InputValidationError("Invalid create job payload")
            raw_path = str(item.get("path", "")).strip()
            if not raw_path:
                raise InputValidationError("Invalid create job payload")
            path = Path(raw_path).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise InputValidationError("Invalid create job payload")
            paths.append(path)
        return paths

    def _build_batch_request(self, payload: dict[str, Any]) -> tuple[BatchRequest, str, str]:
        case_id = str(payload.get("case_id", "")).strip()
        policy_name = str(payload.get("policy", "")).strip()
        output_root_raw = str(payload.get("output_root", "")).strip()
        mapping_mode = str(payload.get("mapping_mode", "auto")).strip().lower() or "auto"
        continue_on_error = bool(payload.get("continue_on_error", True))
        config_path = Path(str(payload.get("config_path", "configs/anonymization_profiles.yaml"))).expanduser().resolve()

        if mapping_mode not in {"auto", "always", "never"}:
            raise InputValidationError("Invalid create job payload")
        if not case_id or not policy_name or not output_root_raw:
            raise InputValidationError("Invalid create job payload")

        output_root = Path(output_root_raw).expanduser().resolve()
        report_path = Path(str(payload.get("report_path") or (output_root / "report.json"))).expanduser().resolve()
        inputs = self._parse_input_paths(payload.get("inputs"))
        policy = load_policy(policy_name, config_path)
        network_mode = "online" if policy.allow_network else "offline"
        request = BatchRequest(
            case_id=case_id,
            policy_name=policy_name,
            policy=policy,
            input_paths=inputs,
            output_root=output_root,
            report_path=report_path,
            mapping_mode=mapping_mode,
            continue_on_error=continue_on_error,
        )
        return request, str(report_path), network_mode

    def create_job(self, payload: dict[str, Any], idempotency_key: str | None = None) -> tuple[int, dict[str, Any]]:
        try:
            request, report_path, network_mode = self._build_batch_request(payload)
        except (InputValidationError, PolicyValidationError, SecurityPolicyError) as exc:
            return self._error(422, exc.code, exc.message_safe)

        digest = self._payload_digest(payload)
        if idempotency_key:
            existing = self._idempotency_index.get(idempotency_key)
            if existing:
                existing_job_id, existing_digest = existing
                if existing_digest != digest:
                    return self._error(409, "IDEMPOTENCY_CONFLICT", "Conflicting idempotent request")
                record = self._jobs[existing_job_id]
                return 202, {"job_id": record.job_id, "status": "accepted", "case_id": record.case_id}

        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            status="running",
            payload=payload,
            case_id=request.case_id,
            report_path=report_path,
            network_mode=network_mode,
        )
        self._jobs[job_id] = record

        try:
            result: BatchResult = self._service.anonymize_batch(request)
            record.status = result.status
            record.totals = dict(result.totals)
            record.documents = [doc.__dict__ for doc in result.documents]
            record.report_path = result.report_path
        except AnonymizationError as exc:
            record.status = "failed"
            return self._error(422, exc.code, exc.message_safe)
        except Exception:
            record.status = "failed"
            return self._error(500, "INTERNAL_ERROR", "Internal error")

        if idempotency_key:
            self._idempotency_index[idempotency_key] = (job_id, digest)

        return 202, {"job_id": job_id, "status": "accepted", "case_id": request.case_id}

    def get_job(self, job_id: str) -> tuple[int, dict[str, Any]]:
        record = self._jobs.get(job_id)
        if record is None:
            return self._error(404, "NOT_FOUND", "Unknown job")
        return 200, {
            "job_id": record.job_id,
            "case_id": record.case_id,
            "status": record.status,
            "totals": record.totals,
            "documents": record.documents,
            "report_path": record.report_path,
            "network_mode": record.network_mode,
        }

    def get_document(self, job_id: str, document_id: str) -> tuple[int, dict[str, Any]]:
        status_code, job = self.get_job(job_id)
        if status_code != 200:
            return status_code, job
        for doc in job.get("documents", []):
            if doc.get("document_id") == document_id:
                return 200, doc
        return self._error(404, "NOT_FOUND", "Unknown document")

    def resolve_mapping(
        self,
        *,
        case_id: str,
        document_id: str,
        mapping_path: str,
        internal_api_key: str | None,
    ) -> tuple[int, dict[str, Any]]:
        expected = os.environ.get("ANON_INTERNAL_API_KEY", "").strip()
        provided = (internal_api_key or "").strip()
        if not expected or not provided or expected != provided:
            return self._error(403, "FORBIDDEN", "Access denied")

        artifact_path = Path(mapping_path).expanduser().resolve()
        if not artifact_path.exists():
            return self._error(404, "MAPPING_NOT_FOUND", "Mapping unavailable")

        try:
            mapping = self._mapping_store.read_mapping(
                case_id=case_id,
                document_id=document_id,
                mapping_path=artifact_path,
            )
        except SecurityPolicyError as exc:
            return self._error(422, exc.code, exc.message_safe)
        except Exception:
            return self._error(500, "INTERNAL_ERROR", "Internal error")

        return 200, {"status": "resolved", "mapping": mapping}
