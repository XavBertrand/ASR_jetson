# Contract: Pipeline Text Backend Integration

## Purpose

Define the single allowed backend contract for ASR pipeline text anonymization.

## Canonical Callable

- Required function: `asr_jetson.postprocessing.transformer_anonymizer.run_transformer_anonymization`
- Scope: transcript text anonymization in ASR pipeline flow.

## Input Contract

- `text` (string, required)
- `domain_entities` (map<string, list<string>>, optional)
- `preserve_dates` (boolean, optional with existing default behavior)

## Output Contract

- Tuple:
  - `anonymized_text` (string)
  - `mapping` (object)

## Behavioral Requirements

1. Pipeline text anonymization MUST invoke the canonical callable above.
2. Pipeline text anonymization MUST NOT invoke any parallel text anonymizer backend.
3. Existing document anonymization behaviors for PDF/DOCX/XLSX are out of scope and unchanged.
4. Existing downstream usage of returned mapping/report generation must remain compatible.

## Error Contract

- On backend failure, pipeline must expose explicit actionable failure for the anonymization step.
- Error outputs must remain sanitized (no raw sensitive transcript snippets).
