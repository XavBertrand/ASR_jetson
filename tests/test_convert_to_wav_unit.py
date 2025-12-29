from __future__ import annotations

import builtins
import subprocess
import shutil
from pathlib import Path

import pytest

from asr_jetson.preprocessing import convert_to_wav as conv


@pytest.mark.unit
def test_convert_to_wav_uses_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.mp3"
    output_path = tmp_path / "output.wav"
    input_path.write_text("data", encoding="utf-8")

    def fake_run(cmd, check, stdout, stderr, text):
        Path(cmd[-1]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = conv.convert_to_wav(input_path, output_path)

    assert result == output_path
    assert output_path.exists()


@pytest.mark.unit
def test_convert_to_wav_no_backend_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.mp3"
    input_path.write_text("data", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda _name: None)

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"pydub", "torchaudio", "soundfile"}:
            raise ImportError("forced")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError):
        conv.convert_to_wav(input_path, tmp_path / "out.wav", use_ffmpeg=True)


@pytest.mark.unit
def test_convert_batch_no_files(tmp_path: Path) -> None:
    result = conv.convert_batch(tmp_path, tmp_path, extensions=(".mp4",))
    assert result == []


@pytest.mark.unit
def test_convert_batch_invokes_converter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    (input_dir / "one.mp4").write_text("a", encoding="utf-8")
    (input_dir / "two.mp4").write_text("b", encoding="utf-8")

    calls: list[Path] = []

    def fake_convert(path, out_path, **_kwargs):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("ok", encoding="utf-8")
        calls.append(Path(path))
        return out_path

    monkeypatch.setattr(conv, "convert_to_wav", fake_convert)

    result = conv.convert_batch(input_dir, output_dir, extensions=(".mp4",))

    assert len(result) == 2
    assert len(calls) == 2
    assert all(p.suffix == ".mp4" for p in calls)
