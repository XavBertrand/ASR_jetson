# Implementation Plan: Secure Document Anonymization Hardening

**Branch**: `002-doc-anonymization` | **Date**: 2026-02-25 | **Spec**: `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/spec.md`
**Input**: Feature specification from `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/spec.md` + user integration constraints (brownfield ASR Jetson repo)

## Summary

Add a secure, format-aware document anonymization capability that integrates into the existing ASR repository without breaking current behavior. The design introduces a dedicated `anonymization` module with layered responsibilities (`core`, `parsers`, `detectors`, `renderers`, `storage`, `cli`, `api`) and reuses existing repo patterns for configuration, logging, output paths, pipeline orchestration, and Docker entrypoints. The primary user-facing surface is a stable batch CLI (`asr anonymize ...`) with resilient per-document error isolation, deterministic placeholders by `case_id`, no-network-by-default policy, true PDF redaction, and encrypted reversible mappings.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: existing `transformers`, `gliner`, `python-docx`, `pydantic`; additions `pymupdf`, `openpyxl`, `cryptography`, `lxml`  
**Storage**: local filesystem artifacts under feature output root (`anonymized/`, `reports/`, optional encrypted `mappings/`, minimal `audit/`) with restricted permissions  
**Testing**: `pytest` + fixtures/golden files per format + PDF underlying-text non-recoverability test + Docker CLI smoke test  
**Target Platform**: Linux x86_64 + Linux aarch64 (Jetson) in local and Docker environments  
**Project Type**: brownfield Python package with CLI-first integration and optional internal web API  
**Internal Contracts**: typed internal models for `Document`, `Span`, `Entity`, `Mapping`, `Report`, `Policy`; mapping/report artifacts carry explicit schema versions and compatibility behavior  
**Determinism Scope**: placeholder generation deterministic inside one `case_id`; namespace isolation across different `case_id` values  
**Performance Goals**: satisfy spec NFRs (95% docs <=5 minutes/document up to 50 pages; batch robustness up to 100 docs with actionable per-doc failures)  
**Constraints**: no sensitive telemetry, no network by default, true PDF redaction, resilient batch, bounded runtime limits configurable via profiles  
**Scale/Scope**: batch anonymization of mixed input sets (`pdf/docx/xlsx/txt`) with per-document status and aggregate reporting  
**Degradation Strategy**: NER optional; fallback chain `NER -> regex -> rules`; explicit non-sensitive warnings in reports and CLI output  
**Security/Privacy**: zero sensitive data in logs/traces/exceptions/metrics; no plaintext mappings at rest; externalized keys only  
**Mapping Encryption**: AES-GCM (AEAD) with per-artifact nonce + AAD binding (`case_id`, schema version, key id), key from env/keystore provider  
**Network Policy**: outbound network disabled by default; explicit profile opt-in required for any egress-dependent component  
**Temporary Files**: temp workspace under run root (`intermediate/tmp`) with immediate cleanup on success/failure and cleanup-failure warning events  
**Storage Permissions**: directories `0700`, sensitive files (`mappings`, reports with reversible pointers) `0600`  
**Format Redaction Coverage**: PDF true redaction; DOCX body/comments/tracked changes/metadata; XLSX visible+hidden sheets/comments/sheet names/formulas; TXT deterministic rewrite  
**Reproducibility**: pinned dependencies (`uv.lock`) + pinned Docker bases/apt packages + deterministic CLI behavior from `case_id`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Research Gate

- [x] **Clarity & Separation**: responsibilities split into dedicated anonymization layers; existing ASR modules remain focused.
- [x] **Typed Contracts**: `Document`, `Span`, `Entity`, `Mapping`, `Report`, `Policy` contract impacts identified with explicit schema versioning.
- [x] **Compatibility**: mapping/report schema versioning and backward-compatible read path are planned.
- [x] **Determinism Scope**: deterministic placeholders tied to `case_id`; cross-case isolation required.
- [x] **Tests & Fixtures**: critical modules and format fixtures/golden outputs are explicitly in scope.
- [x] **Verification Standards**: PDF underlying-text absence test included.
- [x] **Controlled Degradation**: regex/rules fallback and warning behavior included.
- [x] **Performance Limits**: configurable batch/page/size/time limits included.
- [x] **Security & Reproducibility**: env/keystore secrets and reproducible Docker/dependency strategy included.
- [x] **Security Guarantees**: zero sensitive telemetry, AEAD mappings, no-network-by-default included.
- [x] **Data Handling Hardening**: temp-file cleanup and least-privilege storage permissions included.
- [x] **Document Redaction Guarantees**: PDF true redaction and DOCX/XLSX metadata coverage included.
- [x] **Integration Ergonomics**: stable batch CLI, internal API surface, and actionable non-sensitive errors included.

**Gate Result (Pre-Research)**: PASS

## Project Structure

### Documentation (this feature)

```text
/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   |-- anonymization-api.yaml
|   |-- anonymization-cli.md
|   `-- document-anonymizer-service.md
`-- tasks.md (Phase 2 output, not created by this command)
```

### Source Code (repository root)

```text
/home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/
|-- anonymization/
|   |-- core/
|   |   |-- models.py
|   |   |-- interfaces.py
|   |   |-- policy.py
|   |   `-- service.py
|   |-- parsers/
|   |   |-- pdf_parser.py
|   |   |-- docx_parser.py
|   |   |-- xlsx_parser.py
|   |   `-- txt_parser.py
|   |-- detectors/
|   |   |-- regex_detector.py
|   |   |-- ner_detector.py
|   |   `-- rule_detector.py
|   |-- renderers/
|   |   |-- pdf_renderer.py
|   |   |-- docx_renderer.py
|   |   |-- xlsx_renderer.py
|   |   `-- txt_renderer.py
|   |-- storage/
|   |   |-- mapping_store.py
|   |   |-- audit_store.py
|   |   `-- key_provider.py
|   |-- cli/
|   |   `-- anonymize_cli.py
|   `-- api/
|       `-- internal_routes.py
|-- pipeline/
|   |-- full_pipeline.py           # integrate profile-gated doc anonymization hook
|   `-- cli.py                     # preserve existing asr-pipeline behavior
|-- config/
|   `-- config.py                  # extend with anonymization policy/profile settings
`-- postprocessing/
    |-- anonymizer.py              # reuse detectors/utilities where applicable
    `-- transformer_anonymizer.py

/home/xavier/PycharmProjects/ASR_jetson/tests/
|-- unit/anonymization/
|-- integration/anonymization/
|-- data/anonymization/fixtures/
|-- data/anonymization/golden/
`-- smoke/test_anonymize_cli_docker.py

/home/xavier/PycharmProjects/ASR_jetson/docker/
|-- Dockerfile
`-- Dockerfile.jetson
```

**Structure Decision**: Keep the existing monorepo/package structure and add one cohesive `asr_jetson.anonymization` subtree. Reuse existing config/loading, output-path conventions, manifest/report patterns, and CLI orchestration style to avoid duplication and preserve current ASR entrypoints.

## Phase 0: Research Plan

Research outputs are captured in `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/research.md` and resolve all technical clarifications for:
- deterministic placeholder algorithm and scope isolation
- PDF/DOCX/XLSX implementation patterns with true redaction
- key management and AEAD mapping storage
- batch resilience and failure model
- Docker dependency/entrypoint integration without breaking existing usage

## Phase 1: Design and Contracts

Design outputs are captured in:
- `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/data-model.md`
- `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/contracts/`
- `/home/xavier/PycharmProjects/ASR_jetson/specs/002-doc-anonymization/quickstart.md`

These artifacts define:
- entities, validation rules, and state transitions
- stable CLI + internal API + internal service contracts
- brownfield integration points (pipeline, config, logging, storage, Docker)
- configuration and key strategy (`env`/keystore) with no-network-by-default profiles

### Post-Design Gate Re-Check

- [x] All constitution gates remain satisfied after design/contracts updates.
- [x] No unjustified violations identified.

**Gate Result (Post-Design)**: PASS

## Phase 2: Implementation Planning Snapshot

Planned implementation slices (for `tasks.md` generation) are:
1. Core contracts and policy configuration wiring (including schema versioning + migration readers).
2. Parser/detector/renderer path per format with PDF true redaction.
3. Encrypted mapping + audit storage with key-provider abstraction.
4. Stable CLI (`asr anonymize`) and internal API wrapper around `DocumentAnonymizer` service.
5. Pipeline and Docker integration while preserving current defaults/entrypoints.
6. Full test matrix: fixtures, golden outputs, determinism, cross-case isolation, PDF non-recoverability, Docker CLI smoke.

## Complexity Tracking

No constitution violations requiring justification.
