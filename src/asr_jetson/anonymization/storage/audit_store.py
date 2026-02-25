"""Minimal non-sensitive audit event storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asr_jetson.anonymization.core.safe_logging import sanitize_mapping
from asr_jetson.anonymization.storage.fs_security import ensure_dir_permissions, ensure_file_permissions


class AuditStore:
    def __init__(self, output_root: Path, dir_mode: int = 0o700, file_mode: int = 0o600) -> None:
        self.output_root = output_root
        self.dir_mode = dir_mode
        self.file_mode = file_mode
        self.audit_dir = output_root / "audit"
        ensure_dir_permissions(self.audit_dir, self.dir_mode)

    def append(self, case_id: str, event_code: str, trace_id: str, document_id: str | None, safe_metadata: dict[str, Any]) -> Path:
        path = self.audit_dir / f"{case_id}.jsonl"
        sanitized_metadata = sanitize_mapping(safe_metadata)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "document_id": document_id,
            "event_code": event_code,
            "trace_id": trace_id,
            "safe_metadata": sanitized_metadata,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        ensure_file_permissions(path, self.file_mode)
        return path
