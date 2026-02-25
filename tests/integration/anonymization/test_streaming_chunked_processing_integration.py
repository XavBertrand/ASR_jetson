from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main
from asr_jetson.anonymization.core import streaming as streaming_mod


@pytest.mark.integration
def test_large_text_uses_chunked_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large_text = ("Alice Martin alice.martin@example.com\n" * 5000).strip() + "\n"
    source = tmp_path / "large.txt"
    source.write_text(large_text, encoding="utf-8")

    calls: list[int] = []
    original_iter = streaming_mod.iter_text_chunks

    def traced_iter(path: Path, chunk_size: int = streaming_mod.DEFAULT_CHUNK_SIZE):
        for chunk in original_iter(path, chunk_size=chunk_size):
            calls.append(len(chunk))
            yield chunk

    monkeypatch.setattr(streaming_mod, "iter_text_chunks", traced_iter)

    report = tmp_path / "report.json"
    exit_code = anonymize_main(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path),
            "--case-id",
            "CASE-US3-STREAM",
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0
    assert len(calls) > 1
