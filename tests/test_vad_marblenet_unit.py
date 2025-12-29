from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("nemo.collections.asr.models")

from asr_jetson.vad import marblenet


@pytest.mark.unit
def test_marblenet_apply_vad_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    waveform = torch.zeros(1, 16000)
    monkeypatch.setattr(marblenet.torchaudio, "load", lambda _path: (waveform, 16000))

    logits = torch.zeros(1, 10, 2)
    logits[..., 0] = 10.0
    logits[:, 2:5, 0] = 0.0
    logits[:, 2:5, 1] = 10.0

    class DummyModel:
        def __init__(self, logits_tensor):
            self.device = torch.device("cpu")
            self._logits = logits_tensor

        def eval(self):
            return self

        def __call__(self, input_signal=None, input_signal_length=None):
            return self._logits

    model = DummyModel(logits)

    segments = marblenet.apply_vad(
        model,
        wav_path="dummy.wav",
        min_speech_ms=0,
        min_silence_ms=0,
        merge_gap_ms=0,
        pad_ms=0,
        thr_on=0.55,
        thr_off=0.45,
    )

    assert len(segments) == 1
    assert segments[0]["start"] == pytest.approx(0.2)
    assert segments[0]["end"] == pytest.approx(0.5)
