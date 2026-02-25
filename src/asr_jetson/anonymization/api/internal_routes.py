"""Internal API scaffolding for anonymization job lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class JobRecord:
    job_id: str
    status: str
    payload: dict[str, Any]
    documents: list[dict[str, Any]] = field(default_factory=list)


class InternalJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}

    def create(self, job_id: str, payload: dict[str, Any]) -> JobRecord:
        record = JobRecord(job_id=job_id, status="accepted", payload=payload)
        self._jobs[job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)
