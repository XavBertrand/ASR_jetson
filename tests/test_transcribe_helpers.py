from __future__ import annotations

import pytest

from asr_jetson.asr import transcribe


@pytest.mark.unit
def test_attach_speakers_no_diarization() -> None:
    asr_segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
    labeled = transcribe.attach_speakers([], asr_segments)
    assert labeled[0]["speaker"] == 0


@pytest.mark.unit
def test_attach_speakers_single_speaker() -> None:
    diar_segments = [{"start": 0.0, "end": 2.0, "speaker": 3}]
    asr_segments = [{"start": 0.5, "end": 1.0, "text": "hello"}]
    labeled = transcribe.attach_speakers(diar_segments, asr_segments)
    assert all(seg["speaker"] == 3 for seg in labeled)


@pytest.mark.unit
def test_attach_speakers_handles_reversed_diar_segments() -> None:
    diar_segments = [
        {"start": 2.0, "end": 1.0, "speaker": 1},
        {"start": 3.0, "end": 4.0, "speaker": 2},
    ]
    asr_segments = [{"start": 1.1, "end": 1.4, "text": "hello"}]
    labeled = transcribe.attach_speakers(diar_segments, asr_segments)
    assert labeled[0]["speaker"] == 1


@pytest.mark.unit
def test_attach_speakers_gap_fallback() -> None:
    diar_segments = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 2.0, "end": 3.0, "speaker": 1},
    ]
    asr_segments = [{"start": 1.7, "end": 1.8, "text": "gap"}]
    labeled = transcribe.attach_speakers(diar_segments, asr_segments, gap_tol=0.3)
    assert labeled[0]["speaker"] == 1


@pytest.mark.unit
def test_text_by_diar_window() -> None:
    diar_segments = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 1.0, "end": 2.0, "speaker": 1},
    ]
    asr_segments = [
        {"start": 0.2, "end": 0.8, "text": "bonjour"},
        {"start": 1.2, "end": 1.5, "text": "salut"},
    ]
    merged = transcribe.text_by_diar_window(diar_segments, asr_segments)
    assert merged[0]["text"] == "bonjour"
    assert merged[1]["text"] == "salut"
