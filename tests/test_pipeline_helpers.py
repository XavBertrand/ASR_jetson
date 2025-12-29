from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from asr_jetson.pipeline import full_pipeline as fp
from asr_jetson.pipeline.full_pipeline import PipelineConfig


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Meeting 01", "Meeting_01"),
        ("  ...  ", "run"),
        ("foo/bar", "foo_bar"),
    ],
)
def test_sanitize_run_component(raw: str, expected: str) -> None:
    assert fp._sanitize_run_component(raw) == expected


@pytest.mark.unit
def test_build_run_id_uses_stamp() -> None:
    run_id = fp._build_run_id("My File", stamp="20240101T000000Z")
    assert run_id == "My_File_20240101T000000Z"


@pytest.mark.unit
def test_resolve_out_root_is_absolute() -> None:
    cfg = PipelineConfig(out_dir=Path("outputs"))
    out_root = fp._resolve_out_root(cfg)
    assert out_root.is_absolute()
    assert out_root.name == "outputs"


@pytest.mark.unit
def test_resolve_recordings_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ASR_RECORDINGS_ROOT", str(tmp_path))
    cfg = PipelineConfig(out_dir=Path("outputs"))
    resolved = fp._resolve_recordings_root(cfg)
    assert resolved == tmp_path.resolve()


@pytest.mark.unit
def test_find_existing_transcript_picks_latest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    txt_dir = run_root / "txt"
    txt_dir.mkdir(parents=True)
    older = txt_dir / "first.txt"
    newer = txt_dir / "second.txt"
    ignore = txt_dir / "first_anon.txt"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    ignore.write_text("ignore", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    assert fp._find_existing_transcript(run_root) == newer


@pytest.mark.unit
def test_collect_artifacts_skips_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "txt").mkdir(parents=True)
    (run_root / "json").mkdir(parents=True)
    (run_root / "txt" / "a.txt").write_text("a", encoding="utf-8")
    (run_root / "json" / "b.json").write_text("{}", encoding="utf-8")
    (run_root / "manifest.json").write_text("{}", encoding="utf-8")

    artifacts = fp._collect_artifacts(run_root, recordings_root=None)
    names = {item["name"] for item in artifacts}
    categories = {item["category"] for item in artifacts}

    assert names == {"a.txt", "b.json"}
    assert categories == {"txt", "json"}


@pytest.mark.unit
def test_write_manifest_includes_report_and_meta(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    (run_root / "txt").mkdir(parents=True)
    (run_root / "txt" / "out.txt").write_text("hello", encoding="utf-8")

    meta = {
        "meeting_date": "2024-01-01",
        "meeting_report_type": "entretien_collaborateur",
        "uploaded_at": "2024-01-01T10:00:00Z",
        "asr_prompt": "prompt",
        "speaker_context": "context",
        "saved_filename": "audio.wav",
        "original_filename": "orig.wav",
    }
    (run_root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_text("data", encoding="utf-8")

    cfg = PipelineConfig()
    report_outputs = {"report_status": "generated", "report_reason": ""}
    manifest_path = fp._write_manifest(
        run_root,
        run_id="run-1",
        status="ok",
        audio_path=audio_path,
        cfg=cfg,
        report_outputs=report_outputs,
    )

    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1"
    assert data["status"] == "ok"
    assert data["report"]["status"] == "generated"
    assert data["meta"]["meeting_date"] == "2024-01-01"
    assert any(item["name"] == "out.txt" for item in data["artifacts"])


@pytest.mark.unit
def test_resolve_transformers_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fp.torch.cuda, "is_available", lambda: True)
    assert fp._resolve_transformers_device("cuda:2") == 2
    assert fp._resolve_transformers_device("auto") == 0
    assert fp._resolve_transformers_device("cpu") == -1

    monkeypatch.setattr(fp.torch.cuda, "is_available", lambda: False)
    assert fp._resolve_transformers_device("cuda") == -1
    assert fp._resolve_transformers_device("0") == 0
