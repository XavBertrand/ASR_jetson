from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.anonymization.api.internal_routes import InternalJobStore


def _payload(tmp_path: Path, case_id: str = "CASE-IDEMPOTENT") -> dict[str, object]:
    input_file = Path("tests/data/anonymization/fixtures/us1/sample.txt").resolve()
    output_root = (tmp_path / case_id).resolve()
    return {
        "case_id": case_id,
        "policy": "strict_offline",
        "output_root": str(output_root),
        "report_path": str(output_root / "report.json"),
        "mapping_mode": "never",
        "continue_on_error": True,
        "inputs": [{"document_id": "doc-1", "path": str(input_file), "format": "txt"}],
    }


@pytest.mark.integration
def test_idempotency_same_key_same_payload_returns_same_job(tmp_path: Path) -> None:
    store = InternalJobStore()
    payload = _payload(tmp_path)

    status_1, body_1 = store.create_job(payload, idempotency_key="idem-key-1")
    status_2, body_2 = store.create_job(payload, idempotency_key="idem-key-1")

    assert status_1 == 202
    assert status_2 == 202
    assert body_1["job_id"] == body_2["job_id"]


@pytest.mark.integration
def test_idempotency_same_key_different_payload_returns_conflict(tmp_path: Path) -> None:
    store = InternalJobStore()
    payload_a = _payload(tmp_path, case_id="CASE-IDEMPOTENT-A")
    payload_b = _payload(tmp_path, case_id="CASE-IDEMPOTENT-B")

    status_1, _ = store.create_job(payload_a, idempotency_key="idem-key-2")
    status_2, body_2 = store.create_job(payload_b, idempotency_key="idem-key-2")

    assert status_1 == 202
    assert status_2 == 409
    assert body_2["code"] == "IDEMPOTENCY_CONFLICT"
