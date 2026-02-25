# Quickstart: ASR Transformer Text Backend Unification

## Goal

Verify that ASR pipeline text anonymization uses the canonical backend entrypoint `run_transformer_anonymization()` and does not use a parallel text anonymizer path.

## 1) Environment

```bash
cd /home/xavier/PycharmProjects/ASR_jetson
uv sync --extra dev
```

## 2) Run targeted regression tests

```bash
uv run pytest tests -k "transformer and anonymization"
```

## 3) Run explicit guard tests (when added in implementation)

Expected checks:
- canonical function invocation is observed
- no alternate text anonymizer call is observed

Example command pattern:

```bash
uv run pytest tests/unit tests/integration -k "run_transformer_anonymization or backend_unification"
```

## 4) Validate non-goal stability (PDF/DOCX/XLSX)

Run relevant existing document-format regression tests and confirm unchanged behavior:

```bash
uv run pytest tests/integration/anonymization -k "pdf or docx or xlsx"
```

## 5) Acceptance evidence checklist

- Pipeline text flow calls `run_transformer_anonymization()`.
- No parallel text anonymizer path is invoked.
- Relevant existing PDF/DOCX/XLSX tests remain passing.
