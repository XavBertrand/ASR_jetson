from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - optional dependency
    import language_tool_python  # type: ignore

    _HAS_LANGUAGETOOL = True
    _LANGUAGETOOL_IMPORT_ERROR: Optional[Exception] = None
except Exception as _err:  # pragma: no cover - executed when language_tool_python missing
    language_tool_python = None  # type: ignore
    _HAS_LANGUAGETOOL = False
    _LANGUAGETOOL_IMPORT_ERROR = _err


_LT_TOOL: Optional[Any] = None
_LT_INIT_DONE = False
_LT_INIT_ERROR: Optional[str] = None


def _lt_endpoint() -> Optional[str]:
    return os.getenv("LT_ENDPOINT", "").strip() or None


def _lt_disabled() -> bool:
    raw = (os.getenv("DISABLE_LANGUAGETOOL") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _lt_download_version() -> str:
    # ``latest`` snapshots are re-downloaded too often by language_tool_python.
    return (os.getenv("LT_VERSION") or "6.6").strip()


def _prepare_lt_home() -> Optional[str]:
    """
    Ensure a writable LanguageTool cache directory exists and is exported.
    """
    env_value = os.environ.get("LT_HOME") or os.environ.get("LTP_PATH")
    candidates = []
    if env_value:
        candidates.append(Path(env_value))
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            candidates.append(Path(xdg_cache) / "language_tool_python")
        candidates.append(Path.home() / ".cache" / "language_tool_python")
        candidates.append(Path.cwd() / ".cache" / "language_tool_python")
        candidates.append(Path(tempfile.gettempdir()) / "language_tool_python")

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        resolved = str(path)
        os.environ["LT_HOME"] = resolved
        os.environ["LTP_PATH"] = resolved
        return resolved
    return None


def ensure_language_tool() -> Optional[Any]:
    """
    Lazily instantiate a single French LanguageTool instance per process.
    """
    global _LT_TOOL, _LT_INIT_DONE, _LT_INIT_ERROR
    if _LT_TOOL is not None:
        return _LT_TOOL
    if _lt_disabled() or _LT_INIT_DONE or not _HAS_LANGUAGETOOL:
        _LT_INIT_DONE = True
        return None

    _LT_INIT_DONE = True
    try:
        _prepare_lt_home()
        tool_cls = language_tool_python.LanguageTool  # type: ignore[attr-defined]
        endpoint = _lt_endpoint()
        version = _lt_download_version()
        if endpoint:
            _LT_TOOL = tool_cls(
                "fr",
                remote_server=endpoint,
                language_tool_download_version=version,
            )
        else:
            _LT_TOOL = tool_cls("fr", language_tool_download_version=version)
        return _LT_TOOL
    except Exception as err:
        _LT_INIT_ERROR = str(err)
        _LT_TOOL = None
        print(f"⚠️ LanguageTool unavailable: {err}")
        return None
