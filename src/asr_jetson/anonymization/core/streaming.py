"""Streaming and bounded-memory helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator
import zipfile

from asr_jetson.anonymization.core.errors import InputValidationError

DEFAULT_CHUNK_SIZE = 8192


def iter_text_chunks(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


def read_text_streaming(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    return "".join(iter_text_chunks(path, chunk_size=chunk_size))


def _count_document_units(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        # Keep limit checks deterministic while avoiding full-line buffering for huge files.
        size = max(path.stat().st_size, 1)
        return max(1, math.ceil(size / (50 * 1024)))
    if suffix in {".pdf", ".docx", ".xlsx"}:
        try:
            with zipfile.ZipFile(path, "r") as archive:
                if suffix == ".docx":
                    return max(1, len([n for n in archive.namelist() if n.startswith("word/") and n.endswith(".xml")]))
                if suffix == ".xlsx":
                    return max(1, len([n for n in archive.namelist() if n.startswith("xl/worksheets/") and n.endswith(".xml")]))
        except Exception:
            return 1
    return 1


def enforce_limits(paths: list[Path], max_docs: int, max_total_mb: int, max_pages_per_document: int | None = None) -> None:
    if len(paths) > max_docs:
        raise InputValidationError("Document batch exceeds max_documents_per_batch")
    total_bytes = 0
    for p in paths:
        total_bytes += p.stat().st_size
        if max_pages_per_document is not None and _count_document_units(p) > max_pages_per_document:
            raise InputValidationError("Document exceeds max_pages_per_document")
    if total_bytes > max_total_mb * 1024 * 1024:
        raise InputValidationError("Input size exceeds max_total_input_mb")
