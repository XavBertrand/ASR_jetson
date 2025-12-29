#!/usr/bin/env bash
set -euo pipefail

version="${APP_VERSION:-}"
if [[ -z "${version}" || "${version}" == "dev" ]]; then
    if [[ -f "/workspace/pyproject.toml" ]]; then
        version="$(python - <<'PY'
from pathlib import Path
import re

path = Path("/workspace/pyproject.toml")
content = path.read_text(encoding="utf-8")
in_project = False
for raw_line in content.splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("[") and line.endswith("]"):
        in_project = line == "[project]"
        continue
    if in_project:
        match = re.match(r'version\s*=\s*["\']([^"\']+)["\']', line)
        if match:
            print(match.group(1).strip())
            break
PY
)"
    fi
fi

if [[ -n "${version}" ]]; then
    versions_dir="${ASR_VERSIONS_DIR:-/data/.asr_versions}"
    mkdir -p "${versions_dir}"
    printf '%s\n' "${version}" > "${versions_dir}/asr_jetson.txt"
fi

exec "$@"
