# Quickstart: Secure Document Anonymization Hardening

## 1) Install dependencies

```bash
cd /home/xavier/PycharmProjects/ASR_jetson
uv sync --extra dev
```

## 2) Configure profiles and key source

Profile config path:
- `/home/xavier/PycharmProjects/ASR_jetson/configs/anonymization_profiles.yaml`

Default profile behavior:
- `strict_offline`: `allow_network: false`
- `online_opt_in`: explicit network opt-in when required

Key provider settings (for encrypted reversible mappings):

```bash
export ANON_KEY_PROVIDER=env
export ANON_KEY_ID=anon-key-v1
export ANON_MAPPING_KEY="<base64-encoded-aead-key>"
```

Optional internal authorization credential for mapping resolution:

```bash
export ANON_INTERNAL_API_KEY="<internal-shared-secret>"
```

## 3) Run batch anonymization CLI

```bash
uv run asr anonymize \
  --input /data/incoming \
  --output /data/outgoing \
  --case-id CASE-2026-0001 \
  --policy strict_offline \
  --report /data/outgoing/report.json
```

Optional mapping mode:

```bash
uv run asr anonymize \
  --input /data/incoming \
  --output /data/outgoing \
  --case-id CASE-2026-0001 \
  --policy strict_offline \
  --report /data/outgoing/report.json \
  --mapping always
```

Expected outputs:
- `/data/outgoing/anonymized/` anonymized documents
- `/data/outgoing/report.json` batch report
- `/data/outgoing/mappings/` encrypted mapping artifacts (when enabled)
- `/data/outgoing/audit/<case_id>.jsonl` minimal safe audit events

## 4) Mapping resolution authorization boundary

Resolution is restricted to trusted internal callers and requires internal credential:

```bash
uv run asr anonymize \
  --resolve-mapping /data/outgoing/mappings/<doc_id>.enc.json \
  --case-id CASE-2026-0001 \
  --resolve-document-id <document_id> \
  --internal-api-key "$ANON_INTERNAL_API_KEY"
```

Failure behavior:
- missing/invalid internal key => sanitized denial (`SECURITY_POLICY_ERROR` / forbidden in API facade)
- missing mapping => sanitized not-found behavior
- key mismatch/tamper => sanitized security error

## 5) Internal API contract usage notes

Idempotent job creation behavior (`Idempotency-Key`):
- same key + same payload => same `job_id` returned
- same key + different payload => `409 IDEMPOTENCY_CONFLICT`

Contract tests:

```bash
uv run pytest tests/contract/test_anonymization_api_contract.py
uv run pytest tests/contract/test_anonymization_api_idempotency_contract.py
uv run pytest tests/contract/test_anonymization_mapping_auth_contract.py
```

## 6) Docker integration (standard + Jetson)

Build images:

```bash
docker build -t asr-jetson:dev -f docker/Dockerfile .
docker build -t asr-jetson:jetson -f docker/Dockerfile.jetson .
```

Capture image digests for reproducibility evidence:

```bash
docker image inspect asr-jetson:dev --format '{{.Id}}'
docker image inspect asr-jetson:jetson --format '{{.Id}}'
```

Smoke command (same CLI contract in container):

```bash
docker run --rm -v "$PWD/tests/data/anonymization/fixtures/us1:/data/in" -v "$PWD/outputs:/data/out" \
  asr-jetson:dev \
  asr anonymize --input /data/in --output /data/out --case-id CASE-DOCKER-1 --policy strict_offline --report /data/out/report.json --mapping never
```

Pytest smoke (set image tag explicitly):

```bash
ASR_DOCKER_IMAGE=asr-jetson:dev uv run pytest tests/smoke/test_anonymize_cli_docker.py
```

## 7) Standard operating load + benchmark (NFR-001)

Standard operating load:
- Hardware: 8 CPU cores, 32 GB RAM
- Concurrency: single anonymization job
- Dataset: 50 docs mixed formats, total <= 200 MB
- Policy: `strict_offline`, resilient batch

Run benchmark:

```bash
uv run pytest -s tests/perf/test_anonymization_nfr001_benchmark.py
```

Pass criterion: p95 per-document anonymization time < 5 minutes.

## 8) Release gate commands

```bash
tests/acceptance/run_us1_gate.sh
tests/acceptance/run_us2_gate.sh
tests/acceptance/run_us3_gate.sh
```

## 9) Validation commands for this feature

```bash
uv run pytest tests/unit/anonymization
uv run pytest tests/integration/anonymization
uv run pytest tests/contract
uv run pytest tests/perf/test_anonymization_nfr001_benchmark.py
```
