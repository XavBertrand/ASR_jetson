# Tasks: Secure Document Anonymization Hardening

**Input**: Design documents from `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/`
**Prerequisites**: `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/plan.md`, `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/spec.md`, `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/research.md`, `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/data-model.md`, `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/contracts/`

**Tests**: Tests are REQUIRED for touched critical modules and format coverage per spec TR-001..TR-005.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create base anonymization module structure and wiring aligned with brownfield repo patterns.

- [ ] T001 Create anonymization package scaffolding in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/__init__.py
- [ ] T002 Create subpackage initializers for layered architecture in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/__init__.py
- [ ] T003 [P] Add anonymization dependency set and CLI script entry in /home/xavier/PycharmProjects/ASR_jetson/pyproject.toml
- [ ] T004 [P] Add centralized anonymization profile configuration file in /home/xavier/PycharmProjects/ASR_jetson/configs/anonymization_profiles.yaml
- [ ] T005 [P] Add schema version constants for mapping/report artifacts in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/schema.py
- [ ] T006 [P] Add safe anonymization error code definitions in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/errors.py
- [ ] T007 [P] Create anonymization test and fixture directory structure in /home/xavier/PycharmProjects/ASR_jetson/tests/data/anonymization/.gitkeep

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement core contracts, policy, and service backbone required by all user stories.

**⚠️ CRITICAL**: No user story implementation starts before this phase is complete.

- [X] T008 Implement typed core models (Policy/Case/Document/Span/Entity/Mapping/Report/AuditEvent) in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/models.py
- [X] T009 Implement parser/detector/renderer/storage/service interfaces in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/interfaces.py
- [X] T010 Implement policy loading and runtime limit validation in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/policy.py
- [X] T011 Implement DocumentAnonymizer orchestration skeleton and batch status model in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/service.py
- [X] T012 [P] Implement safe logging helpers that block sensitive payloads in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/safe_logging.py
- [X] T013 [P] Implement temporary workspace lifecycle manager in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/tempfiles.py
- [X] T014 [P] Implement filesystem permission utilities (`0700` dirs / `0600` files) in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/storage/fs_security.py
- [X] T015 Implement schema compatibility readers for mapping/report versions in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/storage/compat.py
- [X] T016 Implement initial CLI command scaffold (`asr anonymize`) in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/cli/anonymize_cli.py
- [X] T017 Implement initial internal API scaffold for job lifecycle in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/api/internal_routes.py
- [X] T018 Add foundational policy/error contract tests in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/anonymization/test_policy_and_errors_unit.py

**Checkpoint**: Foundation ready. User stories can now proceed.

---

## Phase 3: User Story 1 - Safe Multi-Format Anonymization (Priority: P1) 🎯 MVP

**Goal**: Deliver secure anonymization for PDF/DOCX/XLSX/TXT with true PDF redaction and stable batch CLI flow.

**Independent Test**: Submit one PDF, one DOCX, and one XLSX with known PII; verify anonymized outputs preserve usable structure and PDF original text is not recoverable.

**Business Structure Preservation Checks (measurable)**:
- DOCX: section count and paragraph count remain within ±5% (excluding removed PII tokens), and document opens without repair prompts.
- XLSX: sheet count preserved; hidden sheets remain hidden; formulas remain valid and workbook opens without errors.
- PDF: page count preserved; document renders without errors; redacted areas do not expose underlying text.

### Tests for User Story 1 (REQUIRED)

- [X] T019 [P] [US1] Add multi-format input fixtures in /home/xavier/PycharmProjects/ASR_jetson/tests/data/anonymization/fixtures/us1/
- [X] T020 [P] [US1] Add golden anonymized outputs for PDF/DOCX/XLSX/TXT in /home/xavier/PycharmProjects/ASR_jetson/tests/data/anonymization/golden/us1/
- [X] T021 [P] [US1] Implement parser unit tests for format text+position extraction in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/anonymization/test_parsers_unit.py
- [X] T022 [P] [US1] Implement renderer unit tests for format rewrite behavior in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/anonymization/test_renderers_unit.py
- [X] T023 [P] [US1] Implement integration test for resilient multi-format batch anonymization in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_multiformat_batch_integration.py
- [X] T024 [P] [US1] Implement PDF underlying-text non-recoverability test in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_pdf_true_redaction_integration.py
- [X] T025 [P] [US1] Implement golden non-regression comparison test in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_golden_outputs_integration.py

### Implementation for User Story 1

- [X] T026 [P] [US1] Implement TXT parser for extracted text/span anchors in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/parsers/txt_parser.py
- [X] T027 [P] [US1] Implement TXT renderer for placeholder application in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/renderers/txt_renderer.py
- [X] T028 [P] [US1] Implement PDF parser with positional anchors for redaction in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/parsers/pdf_parser.py
- [X] T029 [P] [US1] Implement PDF renderer with true redaction apply flow in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/renderers/pdf_renderer.py
- [X] T030 [P] [US1] Implement DOCX parser for body/comments/tracked-changes anchors in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/parsers/docx_parser.py
- [X] T031 [P] [US1] Implement DOCX renderer for body+metadata anonymization in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/renderers/docx_renderer.py
- [X] T032 [P] [US1] Implement XLSX parser for visible/hidden sheet and formula anchors in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/parsers/xlsx_parser.py
- [X] T033 [P] [US1] Implement XLSX renderer for hidden/comment/formula-safe rewrites in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/renderers/xlsx_renderer.py
- [X] T034 [P] [US1] Implement regex detector component in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/detectors/regex_detector.py
- [X] T035 [P] [US1] Implement NER detector component with local model usage in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/detectors/ner_detector.py
- [X] T036 [P] [US1] Implement rule-based detector component in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/detectors/rule_detector.py
- [X] T037 [US1] Integrate parser-detector-renderer flow into DocumentAnonymizer for batch processing in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/service.py
- [X] T038 [US1] Implement stable batch CLI argument contract and output writing in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/cli/anonymize_cli.py
- [X] T039 [US1] Register anonymization CLI without breaking existing pipeline CLI behavior in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/cli.py
- [X] T070 [P] [US1] Implement PDF metadata sanitization integration test (Author/Creator/Producer/XMP removed or replaced safely) in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_pdf_metadata_sanitization_integration.py
- [X] T071 [P] [US1] Implement PDF metadata sanitization in renderer apply flow (safe defaults, no user content) in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/renderers/pdf_renderer.py
- [X] T082 [P] [US1] Implement unit test forcing NER detector unavailability (model missing/disabled) verifying regex fallback and warning code emission in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/anonymization/test_ner_fallback_unit.py
- [X] T083 [P] [US1] Implement integration test simulating NER failure (env flag / dependency missing) verifying degraded status + explicit warning_codes includes NER_UNAVAILABLE and regex-only behavior in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_ner_unavailable_fallback_integration.py

**Checkpoint**: US1 is independently usable as MVP.

---

## Phase 4: User Story 2 - Secure and Deterministic Mapping (Priority: P2)

**Goal**: Deliver deterministic per-case placeholders and encrypted reversible mapping with schema compatibility.

**Independent Test**: Run same input twice with same `case_id` (identical placeholders), then rerun with different `case_id` (isolated placeholders/ids) and verify encrypted mapping is recoverable only through authorized flow.

### Tests for User Story 2 (REQUIRED)

- [X] T040 [P] [US2] Implement deterministic placeholder unit tests for same-case repeatability in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/anonymization/test_placeholders_unit.py
- [X] T041 [P] [US2] Implement integration tests for cross-case namespace isolation in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_case_scope_isolation_integration.py
- [X] T042 [P] [US2] Implement encrypted mapping read/write integration tests in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_mapping_encryption_integration.py
- [X] T072 [P] [US2] Implement mapping tamper-detection test (flip 1 byte in .enc.json => decrypt fails with safe error) in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_mapping_tamper_detection_integration.py
- [X] T043 [P] [US2] Implement schema compatibility tests for mapping/report readers in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/anonymization/test_schema_compatibility_unit.py
- [X] T085 [P] [US2] Implement negative test: mapping resolution denied without required authorization boundary (missing/invalid internal key) with sanitized error in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_mapping_authorization_integration.py

### Implementation for User Story 2

- [X] T044 [P] [US2] Implement HMAC-based placeholder generator scoped by `case_id` in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/placeholders.py
- [X] T045 [P] [US2] Implement key provider abstraction for env/keystore retrieval in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/storage/key_provider.py
- [X] T046 [US2] Implement AES-GCM mapping persistence with AAD metadata binding in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/storage/mapping_store.py
- [X] T047 [US2] Implement mapping/report schema version stamping in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/schema.py
- [X] T048 [US2] Integrate deterministic placeholders and encrypted mapping into batch flow in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/service.py
- [X] T049 [US2] Implement authorized mapping resolution CLI path in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/cli/anonymize_cli.py

**Checkpoint**: US2 is independently testable and secure.

---

## Phase 5: User Story 3 - Controlled Runtime Behavior (Priority: P3)

**Goal**: Enforce no-network default, zero-sensitive telemetry, resilient batch failure handling, secure temp file cleanup, and restricted storage permissions.

**Independent Test**: Execute default and failure-mode runs; verify no outbound network by default, no sensitive logs, secure cleanup of temp artifacts, and least-privilege permissions on outputs.

### Tests for User Story 3 (REQUIRED)

- [X] T050 [P] [US3] Implement integration test for default no-network policy and opt-in override in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_network_policy_integration.py
- [X] T051 [P] [US3] Implement integration test for batch resilience on single-document failure in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_batch_resilience_integration.py
- [X] T052 [P] [US3] Implement integration test for temp-file cleanup on success and failure in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_temp_cleanup_integration.py
- [X] T053 [P] [US3] Implement integration test for storage permission hardening in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_storage_permissions_integration.py
- [X] T054 [P] [US3] Implement unit test for telemetry/log sanitization behavior in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/anonymization/test_safe_logging_unit.py
- [X] T055 [P] [US3] Implement API contract tests for anonymization job endpoints in /home/xavier/PycharmProjects/ASR_jetson/tests/contract/test_anonymization_api_contract.py
- [X] T056 [P] [US3] Implement Docker smoke test for anonymization CLI in /home/xavier/PycharmProjects/ASR_jetson/tests/smoke/test_anonymize_cli_docker.py
- [X] T073 [P] [US3] Implement API idempotency tests for job creation (same Idempotency-Key same payload => same job_id; same key different payload => 409) in /home/xavier/PycharmProjects/ASR_jetson/tests/contract/test_anonymization_api_idempotency_contract.py
- [X] T074 [P] [US3] Implement integration test to verify logs contain no sensitive PII snippets (scan captured logs for unique fixture tokens) in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_log_scrubbing_integration.py
- [X] T076 [P] [US3] Implement integration test for bounded-memory / chunked processing on large inputs (max-total-mb / max-pages) ensuring no full-document load when streaming is available in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_streaming_chunked_processing_integration.py
- [X] T086 [P] [US3] Implement API contract tests enforcing mapping access requires internal authorization (X-Internal-API-Key) in /home/xavier/PycharmProjects/ASR_jetson/tests/contract/test_anonymization_mapping_auth_contract.py
- [X] T087 [P] [US3] Implement integration test for concurrent runs with overlapping filenames across different case_id ensuring output separation and no collisions in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_concurrent_case_filename_collision_integration.py
- [X] T088 [P] [US3] Implement failure-injection test for temp cleanup errors verifying warning + audit event emission (sanitized) in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_cleanup_failure_injection_integration.py
- [X] T089 [P] [US3] Implement integration test for mixed-language + abbreviation PII handling (e.g., FR names + EN email patterns + partial identifiers) verifying correct redaction + warnings where ambiguous in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_mixed_language_abbreviations_integration.py
- [X] T090 [P] [US3] Implement integration test for malformed/corrupted document inputs (corrupt PDF/DOCX/XLSX) verifying safe failure codes, sanitized errors, and batch continuation in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/test_malformed_inputs_integration.py

### Implementation for User Story 3

- [X] T057 [P] [US3] Implement explicit network guard used by detectors/connectors in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/network_guard.py
- [X] T058 [US3] Implement resilient batch status aggregation and degraded warnings in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/service.py
- [X] T059 [P] [US3] Implement minimal non-sensitive audit event storage in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/storage/audit_store.py
- [X] T060 [US3] Implement internal API endpoints from contract in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/api/internal_routes.py
- [X] T061 [US3] Integrate anonymization policy/key env configuration with existing config models in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/config/config.py
- [X] T062 [US3] Integrate optional anonymization hook into existing pipeline without default behavior changes in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py
- [X] T063 [US3] Add anonymization dependencies and non-breaking CLI container invocation support in /home/xavier/PycharmProjects/ASR_jetson/docker/Dockerfile
- [X] T064 [US3] Add Jetson-specific anonymization dependency and invocation support in /home/xavier/PycharmProjects/ASR_jetson/docker/Dockerfile.jetson
- [X] T077 [P] [US3] Implement streaming/chunked processing support where possible (TXT streaming, PDF page-iterator, XLSX row/worksheet iteration) and enforce bounded-memory limits in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/anonymization/core/streaming.py

**Checkpoint**: US3 is independently testable and compliant.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finalize docs, performance checks, and full-system validation.

- [X] T065 [P] Update anonymization usage and security configuration docs in /home/xavier/PycharmProjects/ASR_jetson/README.md
- [X] T066 [P] Add/refresh quickstart verification steps for operators in /home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/quickstart.md
- [X] T067 Validate end-to-end regression run for new anonymization test suites in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/anonymization/
- [X] T068 Validate Docker-based CLI smoke workflow and document command examples in /home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/quickstart.md and /home/xavier/PycharmProjects/ASR_jetson/README.md
- [X] T069 Validate zero-sensitive telemetry and no-network defaults from runtime logs in /home/xavier/PycharmProjects/ASR_jetson/outputs/
- [X] T075 [P] Document API idempotency usage (Idempotency-Key) and error codes (409/422) in /home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/quickstart.md
- [X] T078 [P] Update and verify dependency lockfiles (uv.lock) after adding anonymization deps; add CI check ensuring lock is up-to-date in /home/xavier/PycharmProjects/ASR_jetson/uv.lock and /home/xavier/PycharmProjects/ASR_jetson/.github/workflows/
- [X] T079 [P] Validate reproducible Docker builds (pinned base image digest where feasible + deterministic dependency install) and document exact build command + checksum evidence in /home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/quickstart.md
- [X] T080 [P] Add performance benchmark test for NFR-001 using a fixed fixture dataset + defined hardware profile (CPU/RAM, concurrency) with pass criteria (95% < 5 minutes) in /home/xavier/PycharmProjects/ASR_jetson/tests/perf/test_anonymization_nfr001_benchmark.py
- [X] T081 [P] Document “standard operating load” (hardware/concurrency/dataset) in /home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/quickstart.md and reference it from benchmark docs
- [X] T084 [P] Document mapping authorization boundary (X-Internal-API-Key, failure behavior) in /home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/quickstart.md
- [X] T091 [P] [US1] Add explicit US1 acceptance gate script (run CLI on fixtures, verify golden outputs + PDF non-recoverability) in /home/xavier/PycharmProjects/ASR_jetson/tests/acceptance/run_us1_gate.sh
- [X] T092 [P] [US2] Add explicit US2 acceptance gate script (determinism + cross-case isolation + mapping decrypt with auth) in /home/xavier/PycharmProjects/ASR_jetson/tests/acceptance/run_us2_gate.sh
- [X] T093 [P] [US3] Add explicit US3 acceptance gate script (no-network, no-sensitive-logs, cleanup, permissions, streaming bounds) in /home/xavier/PycharmProjects/ASR_jetson/tests/acceptance/run_us3_gate.sh

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1) has no dependencies.
- Foundational (Phase 2) depends on Setup and blocks all user stories.
- User story phases (Phase 3-5) depend on Foundational completion.
- Polish (Phase 6) depends on completion of selected user stories.

### User Story Completion Order (Dependency Graph)

```text
Phase1 Setup -> Phase2 Foundational -> US1 (P1) -> US2 (P2) -> US3 (P3) -> Phase6 Polish
```

Notes:
- Priority order for delivery is `US1 -> US2 -> US3`.
- After Phase 2, stories are technically parallelizable if staffed, but MVP scope remains US1 first.

### Within Each User Story

- Tests for the story are written before implementation and must fail first.
- Parser/detector/renderer or model/service internals precede CLI/API integration for that story.
- Story checkpoint must pass independently before advancing priority.

## Parallel Execution Examples

### User Story 1

```bash
# Parallel test authoring
T019, T020, T021, T022, T023, T024, T025, T070, T082, T083

# Parallel format implementation
T026, T027, T028, T029, T030, T031, T032, T033

# Parallel detector implementation
T034, T035, T036
```

### User Story 2

```bash
# Parallel testing
T040, T041, T042, T043, T072, T085

# Parallel core components
T044, T045
```

### User Story 3

```bash
# Parallel testing
T050, T051, T052, T053, T054, T055, T056, T073, T074, T076, T086, T087, T088

# Parallel runtime hardening components
T057, T059
```

## Implementation Strategy

### MVP First (Recommended)

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 (US1) only.
3. Validate US1 independently with fixture/golden/PDF non-recoverability tests.
4. Demo/deploy MVP.

### Incremental Delivery

1. Add US2 for deterministic and encrypted mapping.
2. Add US3 for runtime hardening and integration surfaces.
3. Finish with Phase 6 polish and full regression.

### Parallel Team Strategy

1. Team completes Phase 1-2 together.
2. One stream drives US1 MVP.
3. Additional streams implement US2/US3 once foundational interfaces stabilize.
