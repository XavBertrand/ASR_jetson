# Feature Specification: Secure Document Anonymization Hardening

**Feature Branch**: `002-doc-anonymization`  
**Created**: 2026-02-24  
**Status**: Draft  
**Input**: User description: "Renforcer l'anonymisation documentaire avec garanties de sécurité et confidentialité, redaction réelle PDF/DOCX/XLSX, déterminisme des placeholders par case_id, et tests de non-régression."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Safe Multi-Format Anonymization (Priority: P1)

As an operations user, I anonymize business documents and need outputs that remove sensitive
content across supported formats without leaking original data.

**Why this priority**: This is the core value of the feature and the minimum usable release.

**Independent Test**: Submit one PDF, one DOCX, and one XLSX containing known personal data,
then verify anonymized outputs remove the sensitive content in each format and preserve usable
business structure.

**Acceptance Scenarios**:

1. **Given** a PDF containing names and account references, **When** anonymization is run,
   **Then** the resulting PDF no longer contains the original text in underlying content.
2. **Given** a DOCX with body text, comments, and tracked changes containing PII,
   **When** anonymization is run, **Then** all sensitive content in body and metadata is
   anonymized.
3. **Given** an XLSX with hidden sheets, comments, sheet names, and formulas with PII,
   **When** anonymization is run, **Then** sensitive content is anonymized in visible and
   hidden workbook areas.

---

### User Story 2 - Secure and Deterministic Mapping (Priority: P2)

As a compliance user, I need reversible anonymization mappings that are secure and deterministic
within one case while isolated from other cases.

**Why this priority**: Reversible mapping is needed for controlled audit workflows while
maintaining confidentiality guarantees.

**Independent Test**: Run anonymization twice on the same source dataset with the same `case_id`
and compare placeholders; then run with a different `case_id` and verify identifier isolation.

**Acceptance Scenarios**:

1. **Given** the same input and same `case_id`, **When** anonymization runs twice,
   **Then** placeholders are identical across runs.
2. **Given** the same input and a different `case_id`, **When** anonymization runs,
   **Then** placeholders and internal identifiers differ from the first case.
3. **Given** a generated reversible mapping, **When** retrieved through the Authorized Workflow
   defined in "Authorized Workflow for Mapping Resolution",
   **Then** mapping remains protected at rest and usable for de-anonymization.

---

### User Story 3 - Controlled Runtime Behavior (Priority: P3)

As a platform owner, I need secure runtime behavior by default so the anonymization workflow does
not leak data through logs, network calls, temporary files, or weak storage permissions.

**Why this priority**: These controls reduce operational and compliance risk in production.

**Independent Test**: Execute anonymization in default mode and failure mode, then audit logs,
network behavior, temporary artifacts, and output directory permissions.

**Acceptance Scenarios**:

1. **Given** default configuration, **When** anonymization runs,
   **Then** no outbound network call is made unless explicitly enabled.
2. **Given** a processing failure during anonymization, **When** the run terminates,
   **Then** temporary files are removed and no sensitive content appears in logs.
3. **Given** generated outputs and mappings, **When** files are written,
   **Then** restricted permissions are applied to sensitive storage locations.

### Edge Cases

- Input document contains mixed languages, abbreviations, and partial identifiers.
- Input document contains malformed structure (corrupt PDF objects, invalid DOCX package,
  or broken XLSX workbook).
- Optional entity detection is unavailable and processing must continue in degraded mode.
- External service configuration is present but outbound network remains disabled by policy.
- Temporary-file cleanup encounters filesystem errors during exception handling.
- Multiple concurrent cases are processed with overlapping source names.

## Operational Definitions (Reference Only)

This section is reference-only and does not introduce new normative requirements.
The authoritative definitions for load profile, authorized workflow, and remediation are in **Normative Definitions**.

## Normative Definitions

### Standard Operating Load (NFR-001)
Standard operating load is defined as:
- Hardware: 8 CPU cores, 32 GB RAM minimum.
- Concurrency: single anonymization job at a time (no parallel jobs).
- Dataset: 50 documents total, mixed PDF/DOCX/XLSX/TXT, each <= 10 pages (PDF/DOCX) and <= 5 sheets (XLSX), total input <= 200 MB.
- Policy: `strict_offline`, mapping mode `auto`, continue-on-error enabled.

### Authorized Workflow for Mapping Resolution (De-anonymization)
Mapping resolution is permitted only inside the trusted boundary and must enforce:
- Authorization credential required: `X-Internal-API-Key` (or equivalent internal credential), validated server-side.
- Policy must allow mapping resolution and mapping artifact must exist.

Denial behavior (MUST):
- Missing/invalid credential => deny with sanitized error; do not leak existence of mapping or original content.
- Mapping missing => sanitized not-found error.
- Key unavailable/invalid => sanitized security error; do not emit plaintext artifacts.

### Secure Remediation Procedures (Observable)
On security-relevant failures (forbidden network attempt, mapping tamper detected, cleanup failure):
- Emit a sanitized warning code in per-document result and batch report.
- Emit a minimal audit event: timestamp, case_id, document_id (or hash), event_code, trace_id.
- Perform best-effort cleanup; on cleanup failure, emit CLEANUP_FAILED warning + audit event.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST anonymize sensitive content in PDF, DOCX, and XLSX documents.
- **FR-002**: The system MUST perform PDF anonymization compliant with DR-001 (true redaction guarantee).
- **FR-003**: The system MUST perform DOCX anonymization compliant with DR-002.
- **FR-004**: The system MUST perform XLSX anonymization compliant with DR-003.
- **FR-005**: Placeholder generation MUST comply with DET-001.
- **FR-006**: Identifier scoping MUST comply with DET-002.
- **FR-007**: The system MUST generate reversible mapping artifacts that remain protected at rest.
- **FR-008**: The system MUST prevent sensitive document text, snippets, and detected entities
  from appearing in logs, traces, exceptions, and metrics.
- **FR-009**: The system MUST disable outbound network calls by default and allow explicit
  opt-in activation through configuration.
- **FR-010**: The system MUST remove temporary processing files immediately after success or
  failure.
- **FR-011**: The system MUST enforce restricted access permissions on anonymized and mapping
  storage locations.
- **FR-012**: The system MUST emit explicit degraded-mode warnings when optional entity
  detection is unavailable and indicate which outputs may have reduced quality.
- **FR-013**: The system MUST support batch anonymization processing for multiple input files in
  one execution request.

### Non-Functional Requirements

- **NFR-001**: At least 95% of supported documents within the dataset profile defined in
  "Standard Operating Load" MUST complete anonymization within 5 minutes per document
  under that load profile.
- **NFR-002**: End-to-end runs MUST leave zero recoverable temporary artifacts in configured
  temporary directories.
- **NFR-003**: Security validation runs MUST detect zero sensitive-content leaks in telemetry
  outputs.
- **NFR-004**: Determinism validation MUST show 100% placeholder consistency for repeated runs
  with identical inputs and `case_id`.
- **NFR-005**: Regression suites for supported formats MUST pass before release.

### Interface & Data Contracts *(mandatory when data is exchanged between modules)*

- This feature updates internal exchange contracts for `Document`, `Span`, `Entity`,
  `Mapping`, and `Report` to carry deterministic scope and security metadata.
- Mapping and report artifacts MUST include explicit schema version metadata.
- Contract changes MUST define backward compatibility behavior for downstream readers.

### Compatibility & Migration *(mandatory)*

- **CM-001**: Any mapping/report format change MUST increment schema version metadata.
- **CM-002**: Existing mapping/report consumers MUST receive a documented backward-compatible
  read path or migration step.
- **CM-003**: Breaking compatibility MUST fail with an explicit actionable error.

### Failure Modes & Controlled Degradation *(mandatory)*

- If optional entity detection is unavailable, processing MUST continue with degraded
  anonymization behavior and explicit warnings.
- If outbound network is disabled by policy, features requiring egress MUST fail fast with an
  actionable non-sensitive error.
- If temporary cleanup fails, processing MUST report the cleanup failure and trigger secure
  remediation procedures.

### Test & Fixture Requirements *(mandatory)*

- **TR-001**: Every touched critical module MUST include automated tests.
- **TR-002**: Each supported format MUST include input fixtures and golden anonymized outputs.
- **TR-003**: Regression tests MUST validate no behavior drift for protected anonymization rules.
- **TR-004**: PDF validation tests MUST assert original sensitive text is absent from
  underlying PDF content.
- **TR-005**: Degraded-mode and warning behavior MUST be test-covered.

### Security & Secrets *(mandatory)*

- **SEC-001**: Secrets and key material MUST be provided by environment variables or a
  dedicated secret store.
- **SEC-002**: No secret values or key material may be committed to source control.
- **SEC-003**: Reversible mapping protection MUST use authenticated encryption.
- **SEC-004**: User-facing and operator-facing diagnostics MUST remain free of sensitive
  document content.

### Document Redaction Guarantees *(mandatory when file formats are processed)*

- **DR-001**: PDF anonymization MUST remove underlying sensitive content, not only hide it
  visually.
- **DR-002**: DOCX anonymization MUST include metadata-bearing structures such as comments and
  tracked changes.
- **DR-003**: XLSX anonymization MUST include hidden and metadata-bearing workbook elements.

### Determinism & Scope *(mandatory when placeholders/identifiers are generated)*

- **DET-001**: Placeholder generation MUST be deterministic within one `case_id`.
- **DET-002**: Placeholder and identifier namespaces MUST be isolated across different cases.

### Performance & Resource Constraints *(mandatory)*

- Batch processing MUST support configurable execution limits for files, pages, and total size.
- Processing MUST use bounded-memory behavior for large documents.
- Runtime limits MUST be configurable by operations teams without code changes.

### Assumptions

- Authorized users invoking de-anonymization are managed outside this feature scope.
- Input files are supplied by trusted internal workflows, but content may still be malformed.
- `case_id` is provided by upstream orchestration and remains stable for one dossier.
- Existing report consumers will adopt schema version handling as part of migration rollout.

### Key Entities *(include if feature involves data)*

- **Case**: A processing scope identified by `case_id` that defines deterministic placeholder
  namespace and output grouping.
- **Document**: Source file submitted for anonymization, including format type and content.
- **Anonymized Artifact**: Output document with sensitive content removed while preserving
  business readability.
- **Mapping Record**: Reversible association between original sensitive tokens and placeholders,
  stored in protected form.
- **Processing Report**: Structured summary of anonymization results, warnings, and policy status.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of supported-format test fixtures produce anonymized outputs with no
  original sensitive terms present in final validated content.
- **SC-002**: 100% of repeated runs with identical input and `case_id` generate identical
  placeholders, and 100% of runs with different `case_id` values produce isolated identifiers.
- **SC-003**: 100% of security validation runs show zero sensitive-content leaks in logs,
  traces, exceptions, and metrics.
- **SC-004**: At least 95% of batch jobs up to 100 documents complete without manual
  intervention, with actionable failure reports for the remainder.
