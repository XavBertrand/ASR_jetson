# Phase 0 Research: ASR Transformer Text Backend Unification

## Research Inputs

- Feature spec: `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/spec.md`
- Current pipeline integration: `/home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py`
- Canonical backend function: `/home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/postprocessing/transformer_anonymizer.py`

## Findings

### 1) Canonical Integration Entry Point

- Decision: Use `run_transformer_anonymization()` as the only backend entrypoint for pipeline text anonymization.
- Rationale: The feature requirement explicitly mandates function-level invocation and forbids parallel text anonymizers.
- Alternatives considered:
  - Direct `TransformerAnonymizer(...)` instantiation in pipeline: rejected because it leaves room for divergent call paths.
  - New wrapper utility in another module: rejected because it introduces unnecessary indirection and potential parallel backend drift.

### 2) No Parallel Text Backend Rule

- Decision: Define a regression guard that fails if pipeline text anonymization bypasses `run_transformer_anonymization()`.
- Rationale: Structural tests prevent future regressions during refactor and enforce long-term architectural constraint.
- Alternatives considered:
  - Code review only: rejected because non-automated and error-prone.
  - Naming-convention lint only: rejected because it does not prove runtime call behavior.

### 3) Non-Regression Strategy for PDF/DOCX/XLSX

- Decision: Keep document-format behavior out of implementation scope and verify by running relevant existing tests unchanged.
- Rationale: Explicit non-goal requires zero behavior drift for those formats.
- Alternatives considered:
  - Refactor shared anonymization code touching document paths: rejected due to avoidable risk and out-of-scope impact.
  - Add new document logic in this feature: rejected as scope expansion.

### 4) Error/Logging Behavior

- Decision: Reuse existing pipeline error handling and sanitized logging behavior; do not add new logging payload classes.
- Rationale: Feature is integration-focused and should avoid introducing new sensitive telemetry pathways.
- Alternatives considered:
  - New logging layer dedicated to this feature: rejected as over-engineering for current scope.
  - Silent fallback to alternate backend on failure: rejected because it violates single-backend constraint.

## Clarification Status

All `NEEDS CLARIFICATION` items are resolved for this planning phase.
