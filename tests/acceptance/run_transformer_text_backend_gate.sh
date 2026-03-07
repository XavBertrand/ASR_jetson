#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

uv run pytest -q \
  tests/unit/pipeline/test_backend_error_scrub_unit.py \
  tests/unit/pipeline/test_text_backend_adapter_unit.py \
  tests/unit/pipeline/test_text_backend_contract_unit.py \
  tests/unit/pipeline/test_pipeline_transformer_backend_invocation_unit.py \
  tests/unit/pipeline/test_no_parallel_text_anonymizer_unit.py \
  tests/unit/pipeline/test_text_backend_guardrail_unit.py \
  tests/unit/pipeline/test_disabled_mode_no_backend_call_unit.py \
  tests/unit/pipeline/test_ner_unavailable_regex_fallback_unit.py \
  tests/integration/test_pipeline_transformer_backend_integration.py \
  tests/integration/test_pipeline_speaker_context_transformer_backend_integration.py \
  tests/integration/test_pipeline_ner_unavailable_regex_fallback_integration.py \
  tests/integration/test_pipeline_text_backend_determinism_integration.py \
  tests/integration/test_pipeline_runtime_limit_bounded_integration.py \
  tests/integration/test_pipeline_runtime_exception_sanitized_integration.py \
  tests/contract/test_text_backend_schema_compatibility_contract.py \
  tests/contract/test_text_backend_integration_contract.py
