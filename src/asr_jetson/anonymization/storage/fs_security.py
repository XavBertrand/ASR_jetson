"""Filesystem permission hardening utilities."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_dir_permissions(path: Path, mode: int = 0o700) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, mode)


def ensure_file_permissions(path: Path, mode: int = 0o600) -> None:
    if path.exists():
        os.chmod(path, mode)
