# CLI Contract: Document Anonymization

## Stability Scope

- Command namespace: `asr anonymize`
- Compatibility target: additive to existing `asr-pipeline` behavior (no breaking change)
- Output semantics: deterministic placeholders per `case_id`, resilient batch, non-sensitive logs

## Primary Batch Command

```bash
asr anonymize \
  --input <path|dir> \
  --output <dir> \
  --case-id <id> \
  --policy <name> \
  --report <path>
```

### Required Arguments

- `--input`: input file or directory (`pdf/docx/xlsx/txt`)
- `--output`: output root directory
- `--case-id`: deterministic anonymization scope
- `--policy`: policy/profile name from centralized config
- `--report`: report file path (JSON)

### Optional Arguments

- `--mapping {auto,always,never}` (default `auto`; policy may force)
- `--config <path>` (override policy config path)
- `--fail-fast` (default is resilient mode, continue on per-doc failure)

### Behavioral Guarantees

- network calls disabled unless policy explicitly enables.
- one document failure does not stop full batch in default mode.
- CLI errors are actionable and sanitized (no sensitive text snippets).
- placeholders are deterministic for same `case_id` and isolated across different `case_id`.

## Output Contract

For output root `<output>`:

- `<output>/anonymized/<relative_input_path>`: anonymized documents
- `<output>/report.json` (or `--report` target): batch report with per-document status
- `<output>/mappings/<document_or_case>.enc.json`: encrypted mapping artifact when enabled
- `<output>/audit/<case_id>.jsonl`: minimal safe audit trail

### Report JSON Minimum Fields

- `schema_version`
- `case_id`
- `policy`
- `totals`: `total_documents`, `succeeded`, `failed`, `degraded`
- `documents[]`: `document_id`, `status`, `warning_codes[]`, `failure_code`, `failure_message_safe`, `output_path`, `mapping_path`

## Exit Codes

- `0`: all documents succeeded
- `10`: batch completed with some document failures/degraded results
- `20`: policy/config validation failure
- `30`: fatal startup/runtime error before processing
- `40`: security policy violation (forbidden network, invalid key, etc.)

## Docker Invocation Contract

The command must be runnable in existing image without changing current default behavior:

```bash
docker run --rm <image> asr anonymize --input /data/in --output /data/out --case-id CASE-1 --policy strict_offline --report /data/out/report.json
```

This invocation must not require changing existing container ENTRYPOINT defaults.
