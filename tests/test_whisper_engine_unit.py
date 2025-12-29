from __future__ import annotations

import sys
import types

import pytest

from asr_jetson.asr import whisper_engine


@pytest.mark.unit
def test_load_faster_whisper_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls: list[dict] = []

    class DummyWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str, num_workers: int):
            init_calls.append(
                {
                    "model_name": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "num_workers": num_workers,
                }
            )

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = DummyWhisperModel

    ct = types.ModuleType("ctranslate2")

    def get_supported_compute_types(_device: str):
        raise RuntimeError("no cuda")

    ct.get_supported_compute_types = get_supported_compute_types

    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)

    _model, meta = whisper_engine.load_faster_whisper("tiny", device="cuda", compute_type="float16")

    assert meta["device"] == "cpu"
    assert init_calls[0]["device"] == "cpu"


@pytest.mark.unit
def test_load_faster_whisper_selects_supported_compute_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls: list[dict] = []

    class DummyWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str, num_workers: int):
            init_calls.append(
                {
                    "model_name": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "num_workers": num_workers,
                }
            )

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = DummyWhisperModel

    ct = types.ModuleType("ctranslate2")
    ct.get_supported_compute_types = lambda _device: ["int8", "float32"]

    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)

    _model, meta = whisper_engine.load_faster_whisper("tiny", device="cpu", compute_type="float16")

    assert meta["compute_type"] == "int8"
    assert init_calls[0]["compute_type"] == "int8"


@pytest.mark.unit
def test_load_faster_whisper_falls_back_on_init_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    class DummyWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str, num_workers: int):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("init failed")

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = DummyWhisperModel

    ct = types.ModuleType("ctranslate2")
    ct.get_supported_compute_types = lambda _device: ["int8", "float32"]

    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)

    _model, meta = whisper_engine.load_faster_whisper("tiny", device="cpu", compute_type="int8")

    assert calls["count"] == 2
    assert meta["device"] == "cpu"
    assert meta["compute_type"] == "int8"
