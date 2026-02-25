from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_anonymize_cli_smoke_in_docker(tmp_path: Path) -> None:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        pytest.skip("docker binary unavailable")

    image = os.environ.get("ASR_DOCKER_IMAGE", "").strip()
    if not image:
        pytest.skip("ASR_DOCKER_IMAGE is not set")

    inspect = subprocess.run(
        [docker_bin, "image", "inspect", image],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        pytest.skip(f"docker image not available locally: {image}")

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "sample.txt").write_text(
        "Client: Alice Martin\nEmail: alice.martin@example.com\n",
        encoding="utf-8",
    )

    command = [
        docker_bin,
        "run",
        "--rm",
        "-v",
        f"{tmp_path}:/data",
        image,
        "asr",
        "anonymize",
        "--input",
        "/data/in",
        "--output",
        "/data/out",
        "--case-id",
        "CASE-DOCKER-SMOKE",
        "--policy",
        "strict_offline",
        "--report",
        "/data/out/report.json",
        "--mapping",
        "never",
    ]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)

    assert proc.returncode in {0, 10}
    assert (output_dir / "report.json").exists()
