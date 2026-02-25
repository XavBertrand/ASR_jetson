from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


def _run_case(input_dir: Path, output_dir: Path, case_id: str) -> int:
    report = output_dir / "report.json"
    return anonymize_main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--case-id",
            case_id,
            "--policy",
            "strict_offline",
            "--report",
            str(report),
        ]
    )


@pytest.mark.integration
def test_concurrent_case_runs_with_same_filenames_do_not_collide(tmp_path: Path) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir(parents=True, exist_ok=True)
    source_b.mkdir(parents=True, exist_ok=True)

    content = "Contact Alice Martin <alice.martin@example.com>"
    (source_a / "same-name.txt").write_text(content, encoding="utf-8")
    (source_b / "same-name.txt").write_text(content, encoding="utf-8")

    output_a = tmp_path / "out-a"
    output_b = tmp_path / "out-b"

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(_run_case, source_a, output_a, "CASE-CONCURRENT-A")
        fut_b = pool.submit(_run_case, source_b, output_b, "CASE-CONCURRENT-B")
        exit_a = fut_a.result()
        exit_b = fut_b.result()

    assert exit_a == 0
    assert exit_b == 0

    text_a = (output_a / "anonymized" / "same-name.txt").read_text(encoding="utf-8")
    text_b = (output_b / "anonymized" / "same-name.txt").read_text(encoding="utf-8")

    assert text_a != text_b
    assert "alice.martin@example.com" not in text_a
    assert "alice.martin@example.com" not in text_b
