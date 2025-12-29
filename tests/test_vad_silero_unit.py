from __future__ import annotations

import pytest

from asr_jetson.vad.silero import apply_vad, normalize_segments


@pytest.mark.unit
def test_normalize_segments_merges_and_pads() -> None:
    segments = [
        {"start": 0.5, "end": 1.0},
        {"start": 1.05, "end": 1.3},
    ]
    out = normalize_segments(
        segments,
        merge_gap_ms=140,
        min_speech_ms=100,
        pad_ms=100,
        total_sec=2.0,
    )
    assert len(out) == 1
    assert out[0]["start"] == pytest.approx(0.4)
    assert out[0]["end"] == pytest.approx(1.4)


@pytest.mark.unit
def test_apply_vad_with_custom_utils_no_postprocess() -> None:
    def fake_get_speech_timestamps(_wav, _model, **_kwargs):
        return [{"start": 0.1, "end": 0.2}]

    def fake_read_audio(_path, sampling_rate=16000):
        return [0] * sampling_rate

    utils = (fake_get_speech_timestamps, None, fake_read_audio, None, None)
    segments = apply_vad(model=object(), wav_path="dummy.wav", utils=utils, postprocess=False)
    assert segments == [{"start": 0.1, "end": 0.2}]
