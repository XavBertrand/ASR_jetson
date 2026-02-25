from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.slow
@pytest.mark.integration
def test_anonymization_nfr001_p95_under_5_minutes(tmp_path: Path) -> None:
    durations: list[float] = []

    source = tmp_path / "dataset"
    source.mkdir(parents=True, exist_ok=True)

    for idx in range(20):
        input_file = source / f"doc-{idx:03d}.txt"
        input_file.write_text(
            f"Client {idx}: Alice Martin\\nEmail: alice.martin+{idx}@example.com\\n",
            encoding="utf-8",
        )

        output_root = tmp_path / f"run-{idx:03d}"
        report_path = output_root / "report.json"

        start = time.perf_counter()
        exit_code = anonymize_main(
            [
                "--input",
                str(input_file),
                "--output",
                str(output_root),
                "--case-id",
                f"CASE-PERF-{idx:03d}",
                "--policy",
                "strict_offline",
                "--report",
                str(report_path),
                "--mapping",
                "never",
            ]
        )
        duration = time.perf_counter() - start
        assert exit_code == 0
        durations.append(duration)

    p95 = statistics.quantiles(durations, n=100, method="inclusive")[94]
    avg = statistics.fmean(durations)
    print(f"NFR-001 benchmark: docs={len(durations)} avg_s={avg:.3f} p95_s={p95:.3f}")

    assert p95 < 300.0
