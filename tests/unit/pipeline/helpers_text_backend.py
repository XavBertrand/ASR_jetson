from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from pytest import MonkeyPatch


@contextmanager
def spy_callable(monkeypatch: MonkeyPatch, target_module: Any, name: str):
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original = getattr(target_module, name)

    def _wrapper(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(target_module, name, _wrapper)
    try:
        yield calls
    finally:
        monkeypatch.setattr(target_module, name, original)


def make_fake_backend(result_text: str, mapping: dict[str, Any]) -> Callable[..., tuple[str, dict[str, Any]]]:
    def _fake(*args: Any, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return result_text, mapping

    return _fake
