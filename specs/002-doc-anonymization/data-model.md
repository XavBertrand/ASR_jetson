# Data Model: Secure Document Anonymization Hardening

## Design Principles

- Models are typed and explicit to preserve stable internal contracts.
- Mapping/report payloads include schema versions.
- Deterministic placeholders are scoped by `case_id`.
- No model may carry raw sensitive text in logs/telemetry fields.

## Entities

### 1) Policy

- **Purpose**: runtime profile controlling security, detector behavior, and limits.
- **Fields**:
  - `policy_id` (`str`, required)
  - `description` (`str`, optional)
  - `allow_network` (`bool`, default `false`)
  - `enable_ner` (`bool`, default `true`)
  - `enable_regex` (`bool`, default `true`)
  - `enable_rules` (`bool`, default `true`)
  - `continue_on_error` (`bool`, default `true`)
  - `emit_mapping` (`bool`, default `true`)
  - `mapping_required` (`bool`, default `false`)
  - `max_documents_per_batch` (`int`, required, `>0`)
  - `max_pages_per_document` (`int`, required, `>0`)
  - `max_total_input_mb` (`int`, required, `>0`)
  - `mapping_schema_version` (`str`, required)
  - `report_schema_version` (`str`, required)
  - `storage_permissions_dir` (`str`, default `0700`)
  - `storage_permissions_file` (`str`, default `0600`)
- **Validation Rules**:
  - `allow_network` defaults to `false` and must be explicit when `true`.
  - at least one detector path must be enabled.

### 2) Case

- **Purpose**: deterministic namespace and operational boundary for one anonymization batch.
- **Fields**:
  - `case_id` (`str`, required, immutable)
  - `policy_id` (`str`, required)
  - `created_at` (`datetime`, required)
  - `started_at` (`datetime`, optional)
  - `finished_at` (`datetime`, optional)
  - `status` (`enum[pending,running,completed,completed_with_errors,failed]`, required)
- **Validation Rules**:
  - `case_id` non-empty and stable across the run.

### 3) Document

- **Purpose**: one input/output document unit in a case.
- **Fields**:
  - `document_id` (`str`, required)
  - `case_id` (`str`, required)
  - `input_path` (`str`, required)
  - `format` (`enum[pdf,docx,xlsx,txt]`, required)
  - `size_bytes` (`int`, optional)
  - `page_or_sheet_count` (`int`, optional)
  - `status` (`enum[queued,processing,succeeded,degraded,failed]`, required)
  - `output_path` (`str`, optional)
  - `warning_codes` (`list[str]`, optional)
  - `failure_code` (`str`, optional)
  - `failure_message_safe` (`str`, optional)
- **Validation Rules**:
  - `failure_message_safe` must exclude source snippets/entities.

### 4) Span

- **Purpose**: anchor mapping between extracted text and format-specific document locations.
- **Fields**:
  - `span_id` (`str`, required)
  - `document_id` (`str`, required)
  - `start` (`int`, required)
  - `end` (`int`, required)
  - `anchor_type` (`enum[pdf_quad,docx_xpath,xlsx_cell,txt_offset]`, required)
  - `anchor_ref` (`str`, required)
- **Validation Rules**:
  - `start < end`
  - anchor must be valid for selected format.

### 5) Entity

- **Purpose**: detected sensitive unit linked to a span.
- **Fields**:
  - `entity_id` (`str`, required)
  - `document_id` (`str`, required)
  - `span_id` (`str`, required)
  - `entity_type` (`enum[person,organization,location,email,phone,iban,account,siren_siret,date,url,other]`, required)
  - `normalized_value_hash` (`str`, required)  
    Note: hash/fingerprint only in persisted model; raw value stays in-memory only.
  - `detector_source` (`enum[ner,regex,rules,hybrid]`, required)
  - `confidence` (`float`, optional)
  - `placeholder` (`str`, required)
- **Validation Rules**:
  - `placeholder` deterministic for same (`case_id`, `entity_type`, normalized value).

### 6) Mapping

- **Purpose**: reversible association data, encrypted at rest.
- **Fields**:
  - `mapping_id` (`str`, required)
  - `case_id` (`str`, required)
  - `document_id` (`str`, optional)
  - `schema_version` (`str`, required)
  - `encryption_alg` (`literal[AES-GCM]`, required)
  - `key_id` (`str`, required)
  - `nonce_b64` (`str`, required)
  - `aad_digest` (`str`, required)
  - `ciphertext_b64` (`str`, required)
  - `created_at` (`datetime`, required)
- **Validation Rules**:
  - no plaintext mapping values persisted.
  - key bytes are never serialized in artifacts.

### 7) Report

- **Purpose**: batch and per-document anonymization summary.
- **Fields**:
  - `report_id` (`str`, required)
  - `case_id` (`str`, required)
  - `schema_version` (`str`, required)
  - `generated_at` (`datetime`, required)
  - `totals` (`object`, required: `total_documents`, `succeeded`, `failed`, `degraded`)
  - `documents` (`list[DocumentResult]`, required)
  - `warnings` (`list[str]`, optional)
  - `policy_snapshot` (`object`, required)
- **Validation Rules**:
  - includes degraded-mode indicators when applicable.
  - excludes sensitive source text.

### 8) AuditEvent

- **Purpose**: minimal non-sensitive operational trace.
- **Fields**:
  - `event_id` (`str`, required)
  - `case_id` (`str`, required)
  - `document_id` (`str`, optional)
  - `event_type` (`enum[policy_applied,doc_started,doc_degraded,doc_failed,mapping_written,batch_completed]`, required)
  - `timestamp` (`datetime`, required)
  - `safe_metadata` (`dict[str,str|int|bool]`, required)
- **Validation Rules**:
  - never contains raw document text, entity values, or spans.

## Relationships

- `Policy` 1..* `Case`
- `Case` 1..* `Document`
- `Document` 1..* `Span`
- `Span` 1..* `Entity`
- `Case` 1..* `Mapping`
- `Case` 1..1 `Report` per batch run
- `Case` 1..* `AuditEvent`

## State Transitions

### Case

- `pending -> running -> completed`
- `pending -> running -> completed_with_errors`
- `pending -> running -> failed`

### Document

- `queued -> processing -> succeeded`
- `queued -> processing -> degraded`
- `queued -> processing -> failed`

### Mapping

- `in_memory_only -> encrypted_persisted -> archived_or_deleted`

## Contract Versioning and Migration

- `Mapping.schema_version` and `Report.schema_version` are mandatory.
- Compatible readers must support at least one prior minor schema version.
- Unknown major versions fail with explicit actionable non-sensitive error.
