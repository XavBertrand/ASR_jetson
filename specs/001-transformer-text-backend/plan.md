# Implementation Plan: ASR Transformer Text Backend Unification

**Branch**: `001-transformer-text-backend` | **Date**: 2026-02-25 | **Spec**: `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/spec.md`
**Input**: Feature specification from `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/spec.md`

## Summary

Unify ASR transcript text anonymization behind the single canonical backend entrypoint `run_transformer_anonymization()` and remove/avoid parallel text anonymizer call paths in the ASR pipeline path, while keeping PDF/DOCX/XLSX behavior unchanged. Add regression guards proving the canonical function is invoked.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: Existing `transformers`, `gliner`, `rapidfuzz`, `unidecode` via `src/asr_jetson/postprocessing/transformer_anonymizer.py` (no new dependency introduced)  
**Storage**: Existing pipeline filesystem outputs under run directory (`txt/`, `json/`, `reports/`, `pdf/`)  
**Testing**: pytest (unit + integration + contract + perf + acceptance scripts under tests/acceptance/)  
**Target Platform**: Linux x86_64 and Linux aarch64 (Jetson)  
**Project Type**: Brownfield Python package + CLI pipeline  
**Internal Contracts**: Pipeline text anonymization call contract is normalized to one backend entrypoint; no schema changes for `Document/Span/Entity/Mapping/Report` in this feature  
**Determinism Scope**: Preserve existing deterministic behavior of transformer anonymization outputs for identical transcript input and settings; no new case-id namespace introduced  
**Performance Goals**: Median text anonymization duration regression <= 10% vs current baseline scenario  
**Constraints**: MUST call `asr_jetson.postprocessing.transformer_anonymizer.run_transformer_anonymization`; In **nominal mode**, MUST NOT introduce a parallel text anonymizer backend; In degraded mode, regex-only fallback is explicitly permitted (constitution); MUST NOT change PDF/DOCX/XLSX behavior  
**Scale/Scope**: ASR text anonymization path only (pipeline transcription postprocessing), no document-format behavior expansion  
**Degradation Strategy**:
- If the canonical backend is **unavailable at runtime** (ImportError / initialization failure), the pipeline enters **degraded mode** and MUST:
  1) fall back to **regex-only** anonymization (offline, bounded),
  2) emit an explicit WARNING (`NER_UNAVAILABLE_REGEX_FALLBACK`) in the result/telemetry,
  3) remain deterministic within a case_id and isolated across case_id values,
  4) never leak sensitive input text in logs/errors.
- If the canonical backend is available but fails during execution (exception), the pipeline MUST hard-fail the anonymization step with an explicit actionable **sanitized** error.  
**Security/Privacy**: Preserve existing no-sensitive-content logging behavior in text anonymization flow  
**Mapping Encryption**: Unchanged in this feature (no new mapping storage contract introduced)  
**Network Policy**: Unchanged in this feature (no new outbound network behavior introduced)  
**Temporary Files**: Reuse existing pipeline temp/intermediate lifecycle; no new temp artifact class introduced  
**Storage Permissions**: Reuse existing permissions behavior; no new storage class introduced by this feature  
**Format Redaction Coverage**: Explicitly out of change scope for PDF/DOCX/XLSX (must remain behaviorally identical)  
**Reproducibility**: Existing lockfile + deterministic test guard assertions for backend entrypoint usage

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Research Gate

- [x] **Clarity & Separation**: Change is limited to pipeline text anonymization integration seam.
- [x] **Typed Contracts**: Internal call contract is explicit; no hidden alternate backend path.
- [x] **Compatibility**: No mapping/report schema changes in this feature.
- [x] **Determinism Scope**: Existing deterministic transformer behavior preserved; no new namespace mixing.
- [x] **Tests & Fixtures**: Regression tests added for canonical backend invocation and no-parallel-backend rule.
- [x] **Verification Standards**: Existing format verification remains in place; this feature adds no new format surface.
- [x] **Controlled Degradation**: Failure behavior for backend call remains explicit and actionable.
- [x] **Performance Limits**: Regression bound for anonymization duration is defined.
- [x] **Security & Reproducibility**: No new secret path; lockfile/build process unchanged.
- [x] **Security Guarantees**: No new telemetry exposure path introduced.
- [x] **Data Handling Hardening**: No new sensitive artifact path introduced.
- [x] **Document Redaction Guarantees**: PDF/DOCX/XLSX behavior explicitly non-goal and protected.
- [x] **Integration Ergonomics**: Single integration entrypoint contract and regression guard are defined.

**Gate Result (Pre-Research)**: PASS

## Project Structure

### Documentation (this feature)

```text
/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
└── tasks.md (generated by /speckit.tasks)
```

### Source Code (repository root)

```text
/home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/
├── pipeline/
│   └── full_pipeline.py
├── postprocessing/
│   └── transformer_anonymizer.py
└── anonymization/
    └── (document anonymization modules remain unchanged for text-backend unification scope)

/home/xavier/PycharmProjects/ASR_jetson/tests/
├── unit/
├── integration/
└── data/
```

**Structure Decision**: Keep the existing monorepo structure and limit implementation to pipeline text anonymization call sites and targeted regression tests.

## Phase 0: Outline & Research

Research tasks extracted from Technical Context:

1. Integration pattern task: identify the canonical pipeline call seam to invoke `run_transformer_anonymization()` without direct parallel backend usage.
2. Dependency best-practice task: verify recommended use of the function-level entrypoint vs direct class instantiation for stable backend substitution.
3. Regression guard pattern task: define robust tests proving function invocation and proving no alternate text anonymizer path is used.
4. Non-goal protection task: define verification approach to ensure PDF/DOCX/XLSX behavior remains unchanged.

Phase 0 output:
- `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/research.md`

## Phase 1: Design & Contracts

Design outputs:
- `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/data-model.md`
- `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/contracts/text-backend-integration.md`
- `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/contracts/regression-guard-contract.md`
- `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/quickstart.md`

Agent context update command:
- `.specify/scripts/bash/update-agent-context.sh codex`

### Post-Design Gate Re-Check

- [x] No unresolved `NEEDS CLARIFICATION` in design artifacts.
- [x] Constitution requirements remain satisfied for this feature scope.
- [x] Explicit guardrails ensure no parallel text anonymizer introduction.

**Gate Result (Post-Design)**: PASS

## Phase 2: Implementation Planning Snapshot

Planned implementation slices:

1. Normalize ASR pipeline text anonymization call site to `run_transformer_anonymization()`.
2. Remove or avoid direct alternate text backend invocation path in pipeline text flow.
3. Add regression tests proving canonical function invocation.
4. Add regression tests proving no parallel text anonymizer path is used.
5. Run targeted non-regression verification for unchanged PDF/DOCX/XLSX behavior.

## Complexity Tracking

No constitution violations requiring justification.