# Internal Service Contract: `DocumentAnonymizer`

## Purpose

Provide one shared internal service used by:
- CLI (`asr anonymize`)
- internal API endpoints (future webapp integration in same container)

## Interface

```python
class DocumentAnonymizer:
    def anonymize_batch(self, request: BatchRequest) -> BatchResult: ...
    def anonymize_document(self, request: DocumentRequest) -> DocumentResult: ...
```

## Request Models

### `BatchRequest`

- `case_id: str`
- `policy_name: str`
- `input_paths: list[str]`
- `output_root: str`
- `report_path: str`
- `mapping_mode: Literal["auto", "always", "never"]`
- `continue_on_error: bool`

### `DocumentRequest`

- `case_id: str`
- `policy: Policy`
- `input_path: str`
- `output_path: str`
- `format_hint: Literal["pdf", "docx", "xlsx", "txt"] | None`

## Result Models

### `BatchResult`

- `case_id: str`
- `status: Literal["completed", "completed_with_errors", "failed"]`
- `totals: dict[str, int]`
- `documents: list[DocumentResult]`
- `report_path: str`

### `DocumentResult`

- `document_id: str`
- `status: Literal["succeeded", "degraded", "failed"]`
- `output_path: str | None`
- `mapping_path: str | None`
- `warning_codes: list[str]`
- `failure_code: str | None`
- `failure_message_safe: str | None`

## Behavioral Contract

- Must enforce deterministic placeholders for identical `(case_id, normalized entity value)`.
- Must enforce cross-case namespace isolation.
- Must execute with no-network behavior unless policy opt-in explicitly allows network.
- Must isolate document failures so batch can continue when configured.
- Must emit only non-sensitive diagnostics.
- Key resolution must occur through configured KeyProvider.
- Raw key material must never be passed via method arguments.

## Error Model

- `PolicyValidationError`: invalid or unsafe policy.
- `InputValidationError`: unsupported format, missing file, limit exceeded.
- `ProcessingError`: format processing failure for one document.
- `SecurityPolicyError`: attempted forbidden network/secret usage.

Errors surfaced to CLI/API must be actionable and sanitized.
