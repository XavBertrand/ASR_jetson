#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

uv run pytest -q \
  tests/integration/anonymization/test_pdf_true_redaction_integration.py \
  tests/integration/anonymization/test_pdf_metadata_sanitization_integration.py
uv run pytest -q tests/integration/anonymization/test_multiformat_batch_integration.py
uv run pytest -q tests/integration/anonymization/test_golden_outputs_integration.py
