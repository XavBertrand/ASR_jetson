# Data Model: ASR Transformer Text Backend Unification

## Scope Note

This feature does not introduce a new persisted business schema. It formalizes runtime interaction entities for pipeline text anonymization integration.

## Entities

### 1) TextAnonymizationRequest

- Description: Runtime request sent by ASR pipeline to the canonical text anonymization backend.
- Fields:
  - `text` (string, required): transcript content to anonymize.
  - `domain_entities` (map<string, list<string>>, optional): domain hints reused by transformer backend.
  - `preserve_dates` (boolean, required): date-preservation behavior for text anonymization.
- Validation rules:
  - `text` must be non-empty when anonymization is enabled.
  - `domain_entities` keys must represent supported entity labels when provided.
- Relationships:
  - Produces one `TextAnonymizationResult`.

### 2) TextAnonymizationResult

- Description: Runtime output returned by canonical backend and consumed by pipeline postprocessing.
- Fields:
  - `anonymized_text` (string, required)
  - `mapping` (object, required): backend mapping payload used by downstream report/deanonymization paths.
- Validation rules:
  - `anonymized_text` must be serializable to existing output text artifacts.
  - `mapping` must remain compatible with current downstream report generation usage.
- Relationships:
  - Derived from one `TextAnonymizationRequest`.

### 3) BackendInvocationContract

- Description: Integration contract asserting the runtime callable used by pipeline text flow.
- Fields:
  - `function_path` (string, constant): `asr_jetson.postprocessing.transformer_anonymizer.run_transformer_anonymization`
  - `invoked` (boolean): observable during regression tests via mocking/spying.
  - `fallback_used` (boolean): must remain `false` for alternate text anonymizers in this feature scope.
- Validation rules:
  - `function_path` must match canonical path exactly.
  - `fallback_used` must remain false under standard text anonymization flow.

### 4) RegressionGuardEvidence

- Description: Test evidence that contract constraints are enforced.
- Fields:
  - `canonical_call_observed` (boolean)
  - `parallel_backend_observed` (boolean)
  - `status` (enum: `pass`, `fail`)
- Validation rules:
  - `status=pass` requires `canonical_call_observed=true` and `parallel_backend_observed=false`.

## State Transitions

### Text anonymization invocation lifecycle

1. `not_requested` -> `requested` when pipeline anonymization mode is active.
2. `requested` -> `completed` when canonical backend returns `(anonymized_text, mapping)`.
3. `requested` -> `failed` when canonical backend raises an exception.

### Regression guard lifecycle

1. `pending` -> `pass` when canonical invocation is confirmed and no parallel backend path is used.
2. `pending` -> `fail` when canonical invocation is absent or alternate backend invocation is detected.
