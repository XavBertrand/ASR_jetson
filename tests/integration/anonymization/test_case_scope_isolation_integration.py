from __future__ import annotations

from pathlib import Path

import pytest

from asr_jetson.anonymization.cli.anonymize_cli import main as anonymize_main


@pytest.mark.integration
def test_same_case_is_deterministic_and_cross_case_is_isolated(tmp_path: Path) -> None:
    input_file = Path("tests/data/anonymization/fixtures/us1/sample.txt")

    out_a1 = tmp_path / "run_a1"
    out_a2 = tmp_path / "run_a2"
    out_b = tmp_path / "run_b"

    for out_root, case_id in ((out_a1, "CASE-ISO-A"), (out_a2, "CASE-ISO-A"), (out_b, "CASE-ISO-B")):
        report = out_root / "report.json"
        exit_code = anonymize_main(
            [
                "--input",
                str(input_file),
                "--output",
                str(out_root),
                "--case-id",
                case_id,
                "--policy",
                "strict_offline",
                "--report",
                str(report),
                "--mapping",
                "never",
            ]
        )
        assert exit_code == 0

    text_a1 = (out_a1 / "anonymized" / "sample.txt").read_text(encoding="utf-8")
    text_a2 = (out_a2 / "anonymized" / "sample.txt").read_text(encoding="utf-8")
    text_b = (out_b / "anonymized" / "sample.txt").read_text(encoding="utf-8")

    assert text_a1 == text_a2
    assert text_a1 != text_b
