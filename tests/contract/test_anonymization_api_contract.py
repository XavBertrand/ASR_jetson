from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.anonymization.api.internal_routes import InternalJobStore


def _build_payload(tmp_path: Path) -> dict[str, object]:
    input_file = Path("tests/data/anonymization/fixtures/us1/sample.txt").resolve()
    output_root = (tmp_path / "job-output").resolve()
    return {
        "case_id": "CASE-US3-API-001",
        "policy": "strict_offline",
        "output_root": str(output_root),
        "report_path": str(output_root / "report.json"),
        "mapping_mode": "never",
        "continue_on_error": True,
        "inputs": [{"document_id": "doc-1", "path": str(input_file), "format": "txt"}],
    }


@pytest.mark.integration
def test_create_and_get_job_contract(tmp_path: Path) -> None:
    store = InternalJobStore()
    payload = _build_payload(tmp_path)

    create_status, create_body = store.create_job(payload)
    assert create_status == 202
    assert create_body["status"] == "accepted"
    job_id = create_body["job_id"]

    get_status, get_body = store.get_job(job_id)
    assert get_status == 200
    assert get_body["job_id"] == job_id
    assert get_body["case_id"] == payload["case_id"]
    assert get_body["network_mode"] == "offline"
    assert get_body["totals"]["total_documents"] == 1
    assert len(get_body["documents"]) == 1

    document_id = get_body["documents"][0]["document_id"]
    doc_status, doc_body = store.get_document(job_id, document_id)
    assert doc_status == 200
    assert doc_body["document_id"] == document_id


@pytest.mark.integration
def test_create_job_validation_contract_returns_422(tmp_path: Path) -> None:
    store = InternalJobStore()
    status_code, payload = store.create_job({"case_id": "", "inputs": []})

    assert status_code == 422
    assert payload["code"] in {"INPUT_VALIDATION_ERROR", "POLICY_VALIDATION_ERROR"}
    assert "message_safe" in payload


@pytest.mark.integration
def test_unknown_job_returns_not_found() -> None:
    store = InternalJobStore()

    status_code, body = store.get_job("missing-job")
    assert status_code == 404
    assert body["code"] == "NOT_FOUND"
