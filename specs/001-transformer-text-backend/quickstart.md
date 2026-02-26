# Quickstart: ASR Transformer Text Backend Unification

## Goal

Validate that ASR text anonymization uses the canonical backend function
`asr_jetson.postprocessing.transformer_anonymizer.run_transformer_anonymization`,
without introducing a parallel anonymizer in nominal mode, while preserving PDF/DOCX/XLSX behavior.

## Environment

```bash
cd /home/xavier/PycharmProjects/ASR_jetson
uv sync --extra dev
```

## Test Matrix

| Scope | Command | Expected Evidence |
|-------|---------|-------------------|
| US1 canonical invocation | `uv run pytest -q tests/unit/pipeline/test_pipeline_transformer_backend_invocation_unit.py tests/integration/test_pipeline_transformer_backend_integration.py tests/integration/test_pipeline_speaker_context_transformer_backend_integration.py` | Canonical backend call observed for transcript + speaker-context |
| US1 degraded fallback warning | `uv run pytest -q tests/unit/pipeline/test_ner_unavailable_regex_fallback_unit.py tests/integration/test_pipeline_ner_unavailable_regex_fallback_integration.py` | `NER_UNAVAILABLE_REGEX_FALLBACK` warning emitted in `TextAnonymizationResult.warnings` |
| US1 runtime exception hard-fail (T049) | `uv run pytest -q tests/integration -k "runtime_exception and sanitized"` | Pipeline hard-fails with sanitized actionable error, no raw transcript leak |
| US2 no-parallel-backend guardrails | `uv run pytest -q tests/unit/pipeline/test_no_parallel_text_anonymizer_unit.py tests/unit/pipeline/test_text_backend_guardrail_unit.py tests/contract/test_text_backend_integration_contract.py` | Guard fails on alternate backend path; canonical callable path enforced |
| Determinism + cross-case isolation | `uv run pytest -q tests/integration -k "determinism or cross_case or fallback"` | Same case_id deterministic, different case_id isolated (nominal + fallback) |
| Non-goal regression PDF | `uv run pytest -q tests/integration/anonymization/test_pdf_true_redaction_integration.py tests/integration/anonymization/test_pdf_metadata_sanitization_integration.py` | Existing PDF behavior unchanged |
| Non-goal regression DOCX | `uv run pytest -q tests/integration/anonymization/test_multiformat_batch_integration.py tests/integration/anonymization/test_golden_outputs_integration.py` | Existing DOCX behavior unchanged |
| Non-goal regression XLSX | `uv run pytest -q tests/integration/anonymization/test_multiformat_batch_integration.py tests/integration/anonymization/test_golden_outputs_integration.py` | Existing XLSX behavior unchanged |
| Schema compatibility (no migration impact) | `uv run pytest -q tests/contract` | Schema/version compatibility and contract checks pass |
| Perf regression (NFR-001 protocol) | `uv run pytest -q tests/perf/test_transformer_text_backend_performance_regression.py` | Median runtime <= baseline * 1.10 |

## Acceptance Gates

```bash
tests/acceptance/run_transformer_text_backend_gate.sh
tests/acceptance/run_non_goal_format_regression_gate.sh
```

## Release Gate Command Sequence (US1 + US2 + US3)

```bash
uv run pytest -q tests/unit/pipeline tests/integration/test_pipeline_transformer_backend_integration.py tests/integration/test_pipeline_speaker_context_transformer_backend_integration.py tests/integration/test_pipeline_ner_unavailable_regex_fallback_integration.py tests/integration/test_pipeline_runtime_exception_sanitized_integration.py tests/integration/test_pipeline_text_backend_determinism_integration.py tests/integration/test_pipeline_runtime_limit_bounded_integration.py
uv run pytest -q tests/contract
uv run pytest -q tests/perf/test_transformer_text_backend_performance_regression.py
tests/acceptance/run_transformer_text_backend_gate.sh
tests/acceptance/run_non_goal_format_regression_gate.sh
```
