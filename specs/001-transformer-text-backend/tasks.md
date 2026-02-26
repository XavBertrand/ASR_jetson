# Tasks: ASR Transformer Text Backend Unification

**Input**: Design documents from `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/`  
**Prerequisites**: `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/plan.md`, `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/spec.md`, `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/research.md`, `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/data-model.md`, `/home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/contracts/`

**Tests**: Tests are REQUIRED by spec (TR-001..TR-005) and are included per user story.

**Organization**: Tasks are grouped by user story for independent implementation and validation.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare shared integration scaffolding and reusable test assets.

- [X] T001 Create text backend contract scaffold in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/text_backend_contract.py
- [X] T002 Create text backend adapter scaffold in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/text_backend_adapter.py
- [X] T003 [P] Create shared transcript fixture for backend-invocation tests in /home/xavier/PycharmProjects/ASR_jetson/tests/data/pipeline/text_backend/sample_transcript.txt
- [X] T004 [P] Create shared test helper scaffold for backend spying in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/pipeline/helpers_text_backend.py
- [X] T005 [P] Create acceptance gate script scaffold for backend unification in /home/xavier/PycharmProjects/ASR_jetson/tests/acceptance/run_transformer_text_backend_gate.sh

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement the canonical backend seam used by all user stories.

**⚠️ CRITICAL**: No user story implementation starts before this phase is complete.

- [X] T006 Implement `TextAnonymizationRequest` and `TextAnonymizationResult` typed contracts in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/text_backend_contract.py
- [X] T007 Implement canonical adapter function delegating to `run_transformer_anonymization()` in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/text_backend_adapter.py
- [X] T008 Implement canonical callable path guard constant/checks in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/text_backend_adapter.py
- [X] T009 Integrate adapter seam into pipeline anonymization flow entry in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py
- [X] T010 Add sanitized backend failure handling for adapter invocation in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py
- [X] T011 [P] Add unit tests for adapter output contract in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/pipeline/test_text_backend_adapter_unit.py
- [X] T012 [P] Add unit tests for canonical callable path guard in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/pipeline/test_text_backend_contract_unit.py

**Checkpoint**: Canonical backend seam is ready for story implementation.

---

## Phase 3: User Story 1 - Backend Texte Unique (Priority: P1) 🎯 MVP

**Goal**: Ensure pipeline text anonymization always uses `run_transformer_anonymization()`.

**Independent Test**: Run pipeline text anonymization tests and verify canonical callable is invoked for transcript and speaker-context flows.

### Tests for User Story 1 (REQUIRED)

- [X] T013 [P] [US1] Add unit test proving canonical function call for transcript text flow in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/pipeline/test_pipeline_transformer_backend_invocation_unit.py
- [X] T014 [P] [US1] Add integration test validating anonymized text+mapping outputs via canonical backend in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/test_pipeline_transformer_backend_integration.py
- [X] T015 [P] [US1] Add integration test validating speaker-context anonymization also uses canonical backend in /home/xavier/PycharmProjects/ASR_jetson/tests/integration/test_pipeline_speaker_context_transformer_backend_integration.py

### Implementation for User Story 1

- [X] T016 [US1] Replace direct `TransformerAnonymizer` instantiation path with adapter/canonical function for transcript text in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py
- [X] T017 [US1] Route speaker-context anonymization through the canonical adapter path in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py
- [X] T018 [US1] Remove direct pipeline dependency on alternate text backend call style in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py
- [X] T019 [US1] Preserve existing mapping merge and output artifact behavior after integration in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py

**Checkpoint**: US1 is independently functional and validates canonical backend invocation.

---

## Phase 4: User Story 2 - Régression Contrôlée (Priority: P2)

**Goal**: Enforce automated guardrails against parallel text anonymizer reintroduction.

**Independent Test**: Run regression guard tests and verify failures occur if canonical backend is bypassed.

### Tests for User Story 2 (REQUIRED)

- [X] T020 [P] [US2] Add unit regression test forbidding direct `TransformerAnonymizer` usage in pipeline text flow in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/pipeline/test_no_parallel_text_anonymizer_unit.py
- [X] T021 [P] [US2] Add unit test asserting guard failure when alternate backend path is used in /home/xavier/PycharmProjects/ASR_jetson/tests/unit/pipeline/test_text_backend_guardrail_unit.py
- [X] T022 [P] [US2] Add contract test validating canonical callable path from contract docs in /home/xavier/PycharmProjects/ASR_jetson/tests/contract/test_text_backend_integration_contract.py

### Implementation for User Story 2

- [X] T023 [US2] Enforce single-backend guard in pipeline text anonymization entrypoint in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py
- [X] T024 [US2] Add centralized `_anonymize_text_via_backend(...)` helper to prevent parallel call sites in /home/xavier/PycharmProjects/ASR_jetson/src/asr_jetson/pipeline/full_pipeline.py
- [X] T025 [US2] Implement acceptance gate command for canonical backend + no-parallel-backend assertions in /home/xavier/PycharmProjects/ASR_jetson/tests/acceptance/run_transformer_text_backend_gate.sh

**Checkpoint**: US2 enforces long-term backend unification via automated guardrails.

---

## Phase 5: User Story 3 - Stabilité Multi-Formats Documentaires (Priority: P3)

**Goal**: Prove PDF/DOCX/XLSX behaviors remain unchanged while text backend unification is introduced.

**Independent Test**: Run dedicated non-goal regression gate and confirm all targeted format tests remain passing.

### Tests for User Story 3 (REQUIRED)

- [X] T026 [P] [US3] Non-goal regression (PDF): run doc-anonymization PDF integration tests unchanged. Run:
  uv run pytest -q tests/integration/anonymization -k "pdf"
- [X] T027 [P] [US3] Non-goal regression (DOCX): run doc-anonymization DOCX integration tests unchanged. Run:
  uv run pytest -q tests/integration/anonymization -k "docx"
- [X] T028 [P] [US3] Non-goal regression (XLSX): run doc-anonymization XLSX integration tests unchanged. Run:
  uv run pytest -q tests/integration/anonymization -k "xlsx"
- [X] T029 [P] [US3] Add acceptance gate script for PDF/DOCX/XLSX non-goal stability checks in /home/xavier/PycharmProjects/ASR_jetson/tests/acceptance/run_non_goal_format_regression_gate.sh

### Implementation for User Story 3

- [X] T030 [US3] Add feature-specific non-goal regression gate execution to CI workflow in /home/xavier/PycharmProjects/ASR_jetson/.github/workflows/tests.yml
- [X] T031 [US3] Add targeted pytest marker/selection configuration for non-goal format gates in /home/xavier/PycharmProjects/ASR_jetson/pytest.ini
- [X] T032 [US3] Document non-goal format verification commands for this feature in /home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/quickstart.md

**Checkpoint**: US3 confirms unchanged PDF/DOCX/XLSX behavior with executable gates.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finalize documentation, performance evidence, and release-ready validation flow.

- [X] T033 [P] Add text anonymization performance regression benchmark test (<=10% median regression) in /home/xavier/PycharmProjects/ASR_jetson/tests/perf/test_transformer_text_backend_performance_regression.py
  - Baseline storage: commit a JSON snapshot under `tests/perf/baselines/text_backend_perf_baseline.json`
      containing `{ "machine_class": "...", "median_seconds": <float>, "n_runs": 7 }`.
  - Test compares current median vs stored baseline and fails if (current > baseline * 1.10).
- [X] T034 [P] Update feature quickstart with exact gate command matrix and expected evidence in /home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/quickstart.md
- [X] T035 [P] Update root usage documentation to state canonical text backend rule in /home/xavier/PycharmProjects/ASR_jetson/README.md
- [X] T036 Define release gate command sequence for US1+US2+US3 verification in /home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/quickstart.md
- [X] T037 Validate acceptance scripts are executable and aligned with docs in /home/xavier/PycharmProjects/ASR_jetson/tests/acceptance/run_transformer_text_backend_gate.sh
- [X] T038 Validate contract/spec/plan consistency after implementation and sync notes in /home/xavier/PycharmProjects/ASR_jetson/specs/001-transformer-text-backend/plan.md
- [X] T039 [P] Migration/versioning: prove no schema or migration impact (mapping/report formats unchanged). Paths: specs/001-transformer-text-backend/contracts/*.md, src/asr_jetson/anonymization/core/schema.py, tests/contract/ (run: uv run pytest -q tests/contract)
  - Evidence: contract tests green; no changes to schema version fields; no new migration scripts; document output report schema unchanged.
- [X] T040 [P] Security controls: prove no new secret sources introduced (scan). Run: rg -n "OPENAI_API_KEY|MISTRAL|HF_TOKEN|SECRET|PASSWORD|API_KEY" src tests configs .github docker && fail if matches in committed code (allowlist env var docs only).
  - Evidence: scan output attached in PR (or saved under specs/001-transformer-text-backend/evidence/secret-scan.txt).
- [X] T041 [P] Security controls: prove no sensitive content in logs/errors for unified path. Add unit test asserting scrubbing on backend errors. Target: tests/unit/* (new). Paths: src/asr_jetson/anonymization/core/safe_logging.py and integration wrapper.
  - Run: uv run pytest -q tests/unit -k "safe_logging|scrub|backend_error"
- [X] T042 [P] [US1] Disabled-mode: when anonymization is disabled, MUST NOT call run_transformer_anonymization(). Add test using monkeypatch to assert zero calls. Target: tests/unit or tests/integration for pipeline text flow.
  - Run: uv run pytest -q -k "disabled.*no_call|no_backend_call"
- [X] T043 [P] [US1] Integration end to end: NER unavailable => regex-only fallback + explicit warning + sanitation.
  - Force ImportError/initialization failure in transformer backend
  - Assert: `TextAnonymizationResult.warnings` contains:
    - warning_code == NER_UNAVAILABLE_REGEX_FALLBACK
    - warning_level == WARNING
    - warning_message == "NER unavailable => regex-only fallback"
  - Run: uv run pytest -q tests/integration -k "ner_unavailable|regex_fallback"
- [X] T044 [P] [US2] Determinism + cross-case isolation regression (nominal + fallback):
  - Nominal mode:
    - same input + same case_id => identical output across runs
    - same input + different case_id => different placeholders/mapping
  - Fallback mode (force NER unavailable / ImportError):
    - same input + same case_id => identical output across runs
    - same input + different case_id => different placeholders/mapping
  - Coverage: satisfies "fallback-determinism-cross-case" requirement explicitly.
  Store expected outputs under tests/data/anonymization/golden/text/.
  Run: uv run pytest -q tests/integration -k "determinism|cross_case|fallback"
- [X] T045 [P] [US3] Runtime limits: assert existing runtime/streaming limits still enforced in unified path (bounded processing). Add test with oversized input that must trigger limit behavior (specific error code or warning).
  - Run: uv run pytest -q tests/integration -k "runtime_limit|bounded"
- [X] T047 [P] [US1] unit: import/init failure hook: backend import/init failure triggers regex-only fallback + warning contract.
  - Force ImportError (or init failure) when importing/constructing transformer backend.
  - Assert: fallback path executed, warning_code/message/level present in TextAnonymizationResult.warnings, and sanitized error/log behavior.
  - Assert: determinism within same case_id for fallback (unit-level: repeat twice).
  - Run: uv run pytest -q tests/unit/pipeline/test_ner_unavailable_regex_fallback_unit.py
- [X] T048 [P] [US2] Fallback-only determinism + cross-case isolation:
  - Force backend ImportError/init failure to activate regex-only fallback.
  - Assert: same input + same case_id => identical output across 2 runs.
  - Assert: same input + different case_id => different placeholders/mapping.
  - Assert: warning contract present in TextAnonymizationResult.warnings (code/level/message).
  - Run: uv run pytest -q tests/integration -k "fallback and determinism and cross_case"
- [X] T049 [P] [US1] Integration test: backend runtime exception => hard-fail with actionable sanitized error (no raw transcript).
  - Force `run_transformer_anonymization()` to raise RuntimeError during execution (NOT ImportError/init).
  - Assert: pipeline fails the anonymization step (no regex fallback), and error is actionable + sanitized (no sensitive text).
  - Run: uv run pytest -q tests/integration -k "runtime_exception and sanitized"
---

## Dependencies & Execution Order

### Phase Dependencies

- Phase 1 (Setup): no dependencies.
- Phase 2 (Foundational): depends on Phase 1 and blocks all user stories.
- Phase 3 (US1): depends on Phase 2.
- Phase 4 (US2): depends on Phase 3 for canonical invocation baseline.
- Phase 5 (US3): depends on Phase 3 for stable integrated text flow.
- Phase 6 (Polish): depends on completion of selected user stories.

### User Story Completion Order (Dependency Graph)

```text
Phase1 Setup -> Phase2 Foundational -> US1 (P1) -> US2 (P2)
                                           \-> US3 (P3)
US2 + US3 -> Phase6 Polish
```

### Within Each User Story

- Write tests first and verify they fail before implementation.
- Implement code changes for the story.
- Re-run story-specific tests until pass.
- Confirm independent test criterion before advancing.

## Parallel Execution Examples

### User Story 1

```bash
# Parallel tests
T013, T014, T015
```

### User Story 2

```bash
# Parallel tests
T020, T021, T022
```

### User Story 3

```bash
# Parallel tests
T026, T027, T028, T029
```

## Implementation Strategy

### MVP First (Recommended)

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 (US1) only.
3. Validate canonical backend invocation and no direct alternate text anonymizer usage.
4. Demo/deploy MVP.

### Incremental Delivery

1. Add US2 guardrails and acceptance gate.
2. Add US3 non-goal stability gates for PDF/DOCX/XLSX.
3. Finish Phase 6 polish and performance evidence.

### Parallel Team Strategy

1. Team aligns on foundational seam (Phase 1-2).
2. One stream delivers US1 integration.
3. Additional streams can deliver US2 and US3 in parallel after US1 baseline is stable.
