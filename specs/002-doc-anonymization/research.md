# Research: Secure Document Anonymization Hardening

## Phase 0 Inputs

Feature context: brownfield ASR Jetson repository, with mandatory non-breaking integration and explicit layered architecture:
- `core/anonymization`: interfaces + models (`Entity`, `Span`, `Mapping`, `Report`, `Policy`)
- `parsers`: extract text + positions per format (`pdf/docx/xlsx/txt`)
- `detectors`: regex + NER + rules
- `renderers`: write anonymized documents per format
- `storage`: encrypted mapping + minimal audit
- `cli` + `api`: batch and webapp-ready integration

## Extracted Research Tasks

### Clarification Tasks (from technical context)

- Task: "Research deterministic placeholder strategy for `case_id`-scoped anonymization in ASR document pipeline"
- Task: "Research secure reversible mapping storage format and schema version strategy for ASR anonymization"
- Task: "Research no-network-by-default enforcement pattern compatible with existing ASR runtime"
- Task: "Research resilient batch failure model where one document failure does not stop lot processing"
- Task: "Research Docker integration strategy that adds anonymization dependencies without breaking current container entrypoints"
- Task: "Research stable CLI contract `asr anonymize --input ...` compatible with existing `asr-pipeline` command"

### Dependency Best-Practice Tasks

- Task: "Find best practices for PyMuPDF true redaction in security-sensitive PDF anonymization"
- Task: "Find best practices for `python-docx` + `lxml` to sanitize DOCX comments/tracked changes/metadata"
- Task: "Find best practices for `openpyxl` anonymization of hidden sheets/comments/formulas"
- Task: "Find best practices for `cryptography` AEAD mapping encryption and key management via env/keystore"
- Task: "Find best practices for deterministic hashing/HMAC placeholder generation for privacy-preserving tokens"

### Integration Pattern Tasks

- Task: "Research brownfield integration pattern to reuse existing ASR config/logging/path/pipeline/error handling utilities"
- Task: "Research internal service boundary (`DocumentAnonymizer`) usable by CLI now and webapp later in same container"
- Task: "Research Docker smoke-test approach for CLI integration in CI and local containers"

## Decisions

### 1) Brownfield module placement and reuse

- **Decision**: Add `src/asr_jetson/anonymization/` with submodules `core`, `parsers`, `detectors`, `renderers`, `storage`, `cli`, `api` while reusing existing repository utilities:
  - config patterns from `src/asr_jetson/config/config.py`
  - pipeline run/output conventions from `src/asr_jetson/pipeline/full_pipeline.py`
  - current anonymization detector helpers from `src/asr_jetson/postprocessing/*` where reusable
  - existing CLI and manifest/report output behaviors
- **Rationale**: Meets required layered architecture and avoids duplication while preserving current ASR behavior.
- **Alternatives considered**:
  - Extend only `postprocessing/`: rejected due to mixed concerns and poor separation.
  - Build separate service/repo: rejected due to higher integration/migration overhead.

### 2) True PDF redaction

- **Decision**: Use PyMuPDF redaction annotations + apply-redactions workflow; validate with extraction test proving original text is absent from content stream.
- **Rationale**: Satisfies constitutional true-redaction requirement and explicit PDF non-recoverability test requirement.
- **Alternatives considered**:
  - visual overlay only: rejected because text remains recoverable.
  - image rasterization pipeline: rejected due to quality/accessibility degradation.

### 3) DOCX and XLSX deep coverage

- **Decision**: DOCX anonymization combines high-level `python-docx` edits with `lxml` processing for XML parts (comments, tracked changes, metadata). XLSX anonymization uses `openpyxl` for visible/hidden sheets, comments, sheet names, and formulas.
- **Rationale**: Required to cover metadata-bearing structures beyond body text.
- **Alternatives considered**:
  - body-only edits: rejected as incomplete.
  - conversion to plain text and rebuild: rejected due to structural loss.

### 4) Deterministic placeholder generation

- **Decision**: Generate placeholders from HMAC-SHA256 over `(case_id, normalized_value, entity_type, mapping_schema_version)` with fixed formatting per entity type.
- **Rationale**: deterministic within case, isolated across cases, stable under retries/concurrency.
- **Alternatives considered**:
  - counters only: rejected as order-dependent and unstable.
  - unsalted hash: rejected for weaker privacy guarantees.

### 5) Mapping protection and key management

- **Decision**: Encrypt reversible mappings with AES-GCM (`cryptography`) and AAD binding (`case_id`, schema version, key id). Key material sourced only from env (`ANON_MAPPING_KEY`) or keystore adapter (`ANON_KEY_PROVIDER`, `ANON_KEY_ID`).
- **Rationale**: meets AEAD mandate and externalized secret requirement.
- **Alternatives considered**:
  - plaintext mappings: rejected (security violation).
  - repo-stored key files: rejected (constitution + secret policy violation).

### 6) No-network-by-default runtime policy

- **Decision**: policy profiles default `allow_network=false`; any network-dependent detector/connector is disabled unless explicitly enabled by selected policy.
- **Rationale**: enforces constitution and prevents accidental data egress.
- **Alternatives considered**:
  - opt-out networking: rejected as non-compliant.

### 7) Batch resilience behavior

- **Decision**: per-document isolation with status model (`succeeded`, `failed`, `degraded`), aggregate report at end, and non-zero partial-failure exit code that still returns outputs.
- **Rationale**: one failed document must not abort entire batch.
- **Alternatives considered**:
  - fail-fast batch: rejected by explicit requirement.

### 8) CLI and API integration surface

- **Decision**: introduce stable CLI contract `asr anonymize --input <path|dir> --output <dir> --case-id <id> --policy <name> --report <path>` with optional encrypted mapping output based on policy; expose matching internal API endpoints and shared `DocumentAnonymizer` service callable by future webapp.
- **Rationale**: CLI is mandatory now; API/service enables webapp extension later without redesign.
- **Alternatives considered**:
  - API-only: rejected because standalone CLI is mandatory.
  - CLI-only with no service boundary: rejected because webapp reuse would duplicate logic.

### 9) Docker integration and entrypoints

- **Decision**: extend existing Dockerfiles with anonymization dependencies (including native PDF deps when needed), keep current container entrypoint semantics unchanged, and run CLI via explicit command override (`python -m ...` or console script) in same image.
- **Rationale**: includes feature in existing images without breaking current container workflows.
- **Alternatives considered**:
  - dedicated anonymization image only: rejected due to operational split and drift risk.
  - replacing default entrypoint/CMD: rejected because it may break existing usage.

### 10) Test strategy and fixture discipline

- **Decision**: add fixtures + golden outputs per format (`pdf/docx/xlsx/txt`), determinism/cross-case tests, degraded-mode tests, secure-log tests, and Docker CLI smoke integration.
- **Rationale**: aligns with constitution verification standards and compatibility safeguards.
- **Alternatives considered**:
  - minimal unit tests only: rejected due to regression and security risk.

## Resolved Clarifications

All clarification points are resolved by decisions above.

- No `NEEDS CLARIFICATION` items remain.
- No constitution gate violations identified.
