#!/usr/bin/env bash
set -euo pipefail

cd /home/xavier/PycharmProjects/ASR_jetson

uv run pytest -m "unit or integration" \
  tests/unit/anonymization/test_parsers_unit.py \
  tests/unit/anonymization/test_renderers_unit.py \
  tests/unit/anonymization/test_ner_fallback_unit.py \
  tests/integration/anonymization/test_multiformat_batch_integration.py \
  tests/integration/anonymization/test_pdf_true_redaction_integration.py \
  tests/integration/anonymization/test_golden_outputs_integration.py \
  tests/integration/anonymization/test_pdf_metadata_sanitization_integration.py \
  tests/integration/anonymization/test_ner_unavailable_fallback_integration.py
