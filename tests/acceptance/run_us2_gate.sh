#!/usr/bin/env bash
set -euo pipefail

cd /home/xavier/PycharmProjects/ASR_jetson

uv run pytest -m "unit or integration" \
  tests/unit/anonymization/test_placeholders_unit.py \
  tests/unit/anonymization/test_schema_compatibility_unit.py \
  tests/integration/anonymization/test_case_scope_isolation_integration.py \
  tests/integration/anonymization/test_mapping_encryption_integration.py \
  tests/integration/anonymization/test_mapping_tamper_detection_integration.py \
  tests/integration/anonymization/test_mapping_authorization_integration.py
