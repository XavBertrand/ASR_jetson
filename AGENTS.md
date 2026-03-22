# Repository Guidelines

## Project Structure & Module Organization
- `src/asr_jetson/` holds the core pipeline: `asr/`, `diarization/`, `preprocessing/`, `postprocessing/`, `pipeline/`, and shared `utils/`.
- `configs/` contains runtime YAML presets (e.g., `configs/dev.yaml`, `configs/jetson.yaml`).
- `tests/` contains pytest suites and fixtures; sample inputs live in `tests/data/`.
- `scripts/` holds helper utilities (benchmarking, TensorRT helpers, etc.).
- `docker/` contains Dockerfiles for desktop and Jetson builds.
- `models/` and `outputs/` are local caches/artifacts (typically gitignored).

## Build, Test, and Development Commands
- `uv sync --extra dev` installs dependencies (use `--extra gpu-linux` or `--extra gpu-jetson` for CUDA targets).
- `uv run asr-pipeline --audio path/to/file.wav --out-dir outputs` runs the CLI pipeline locally.
- `uv run pytest` runs the full test suite with coverage (`--cov=src/asr_jetson` is configured).
- `uv run pytest -m "not gpu"` skips GPU-dependent tests when CUDA/Pyannote is unavailable.
- `docker build -t asr-jetson:dev -f docker/Dockerfile .` builds a desktop image; see `docker/Dockerfile.jetson` for Jetson.

## Coding Style & Naming Conventions
- Python 3.11; format with Black (line length 100) and lint with Ruff.
- Prefer `snake_case` for modules/functions and `PascalCase` for classes.
- Test discovery follows `test_*.py` files with `Test*` classes and `test_*` functions.
- Type checking uses mypy; keep public APIs typed when practical.

## Testing Guidelines
- Pytest is the primary framework; use markers `unit`, `integration`, `gpu`, `slow` as appropriate.
- Integration tests may require Hugging Face models; set `HUGGINGFACE_TOKEN` if needed.
- Keep fixtures in `tests/data/` and avoid large binary fixtures in Git history.

## Commit & Pull Request Guidelines
- Recent history uses short, descriptive summaries like `Feature/...` or `(feat) ...` and may include PR numbers (e.g., `(#23)`).
- Keep commit subjects in the imperative, add scope when helpful, and include the issue/PR reference if one exists.
- PRs should describe hardware target (CPU/GPU/Jetson), key model choices, and include test commands run.

## Configuration & Secrets
- `HUGGINGFACE_TOKEN` is required for private Pyannote pipelines.
- `MISTRAL_API_KEY` is required if meeting reports are enabled.

## Active Technologies
- Python 3.11 + existing `transformers`, `gliner`, `python-docx`, `pydantic`; planned additions `pymupdf` (PDF redaction), `openpyxl` (XLSX), `cryptography` (AEAD), `lxml` (DOCX XML metadata edits) (002-doc-anonymization)
- filesystem artifacts under feature-specific output root; encrypted mapping files plus minimal audit events (002-doc-anonymization)
- Python 3.11 + existing `transformers`, `gliner`, `python-docx`, `pydantic`; additions `pymupdf`, `openpyxl`, `cryptography`, `lxml` (002-doc-anonymization)
- local filesystem artifacts under feature output root (`anonymized/`, `reports/`, optional encrypted `mappings/`, minimal `audit/`) with restricted permissions (002-doc-anonymization)
- Python 3.11 + Existing `transformers`, `gliner`, `rapidfuzz`, `unidecode` via `src/asr_jetson/postprocessing/transformer_anonymizer.py` (no new dependency introduced) (001-transformer-text-backend)
- Existing pipeline filesystem outputs under run directory (`txt/`, `json/`, `reports/`, `pdf/`) (001-transformer-text-backend)

## Recent Changes
- 002-doc-anonymization: Added Python 3.11 + existing `transformers`, `gliner`, `python-docx`, `pydantic`; planned additions `pymupdf` (PDF redaction), `openpyxl` (XLSX), `cryptography` (AEAD), `lxml` (DOCX XML metadata edits)
