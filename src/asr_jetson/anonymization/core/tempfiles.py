"""Temporary workspace lifecycle helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


class TempWorkspace:
    def __init__(self, run_root: Path) -> None:
        self._run_root = run_root
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            raise RuntimeError("Temporary workspace not initialized")
        return self._path

    def __enter__(self) -> "TempWorkspace":
        base = self._run_root / "intermediate" / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        self._path = Path(tempfile.mkdtemp(prefix="anon_", dir=base))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def cleanup(self) -> bool:
        if self._path is None:
            return True
        if os.environ.get("ASR_ANON_FORCE_CLEANUP_FAILURE", "").strip().lower() in {"1", "true", "yes"}:
            return False
        try:
            shutil.rmtree(self._path, ignore_errors=False)
            return True
        except OSError:
            return False
