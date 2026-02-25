#!/usr/bin/env bash
set -euo pipefail

cd /home/xavier/PycharmProjects/ASR_jetson

uv run pytest \
  tests/unit/anonymization/test_safe_logging_unit.py \
  tests/integration/anonymization/test_network_policy_integration.py \
  tests/integration/anonymization/test_batch_resilience_integration.py \
  tests/integration/anonymization/test_temp_cleanup_integration.py \
  tests/integration/anonymization/test_storage_permissions_integration.py \
  tests/integration/anonymization/test_log_scrubbing_integration.py \
  tests/integration/anonymization/test_streaming_chunked_processing_integration.py \
  tests/integration/anonymization/test_cleanup_failure_injection_integration.py \
  tests/integration/anonymization/test_mixed_language_abbreviations_integration.py \
  tests/integration/anonymization/test_malformed_inputs_integration.py \
  tests/integration/anonymization/test_concurrent_case_filename_collision_integration.py
