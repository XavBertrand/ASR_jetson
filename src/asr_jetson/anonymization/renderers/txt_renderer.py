"""TXT renderer applying deterministic replacements."""

from __future__ import annotations

from pathlib import Path


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
    for source in sorted(replacements.keys(), key=len, reverse=True):
        text = text.replace(source, replacements[source])
    return text


class TxtRenderer:
    def render(self, input_path: Path, output_path: Path, replacements: dict[str, str]) -> None:
        text = input_path.read_text(encoding="utf-8")
        out = _apply_replacements(text, replacements)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(out, encoding="utf-8")
