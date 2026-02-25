# Contract: Regression Guard for Canonical Text Backend

## Purpose

Define test-level contract that proves backend unification remains enforced.

## Required Regression Assertions

1. Canonical invocation assertion:
   - Regression tests MUST prove `run_transformer_anonymization()` is called during pipeline text anonymization flow.

2. Parallel backend exclusion assertion:
   - Regression tests MUST fail if an alternative text anonymizer path is called in the same flow.

3. Non-goal protection assertion:
   - Relevant existing PDF/DOCX/XLSX tests MUST remain passing without behavior changes introduced by this feature.

## Minimal Test Evidence

- One targeted test with function spy/mock on canonical callable.
- One targeted test or assertion set proving no alternate text backend invocation path.
- Existing relevant regression suite pass result for unchanged format behavior.
