"""High-level transcription helpers built on Faster-Whisper outputs."""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple


def _sec(x_samples: int, sr: int = 16000) -> float:
    """
    Convert a sample count into seconds for a given sampling rate.

    :param x_samples: Number of samples to convert.
    :type x_samples: int
    :param sr: Sampling rate in Hertz.
    :type sr: int
    :returns: Duration in seconds.
    :rtype: float
    """
    return x_samples / float(sr)


# def transcribe_segments(model, wav_path, diar_segments, language=None):
#     """
#     Transcribe using the stable defaults of Faster-Whisper.
#     Audio windows are limited to [start, end] to reduce encoder load.
#     """
#     results = []
#     for seg in diar_segments:
#         start = float(seg.get("start", 0.0))
#         end = float(seg.get("end", 0.0))
#         if end <= start or (end - start) < 0.05:
#             continue
#
#         try:
#             # Window: request the exact duration from Faster-Whisper.
#             segments, info = model.transcribe(
#                 wav_path,
#                 language=language,     # None = auto
#                 task="transcribe",
#                 vad_filter=False,      # VAD already applied upstream
#                 chunk_length=15,       # petit chunk → moins de pression
#                 word_timestamps=False,
#             )
#             text = "".join(s.text for s in segments)
#         except SystemExit:
#             print("[WARN] CTranslate2 aborted during encode(); empty text for this segment.")
#             text = ""
#         except Exception as e:
#             print(f"[WARN] Whisper transcribe failed on [{start:.2f}, {end:.2f}]s: {e}")
#             text = ""
#
#         results.append({**seg, "text": text})
#
#     return results

def transcribe_full(
    model: Any,
    wav_path: str | Path,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Perform a single ASR pass over the entire audio file.

    :param model: Faster-Whisper compatible model instance.
    :type model: Any
    :param wav_path: Path to the WAV audio to transcribe.
    :type wav_path: str | Path
    :param language: Optional forced language code supplied to the decoder.
    :type language: str | None
    :param initial_prompt: Optional context prompt forwarded to Faster-Whisper.
    :type initial_prompt: str | None
    :returns: List of segment dictionaries with ``start``, ``end``, and ``text``.
    :rtype: List[Dict[str, Any]]
    """
    segments, info = model.transcribe(
        wav_path,
        language=language or "fr",  # force French when language is known
        task="transcribe",
        # Let Faster-Whisper apply its internal VAD to reduce looping.
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        # Shorter chunks help reduce drift.
        chunk_length=20,
        word_timestamps=False,

        # Conservative beam search to avoid drift.
        beam_size=5,
        temperature=0.0,  # disable sampling
        # Never condition on previous text (major source of repetitions).
        condition_on_previous_text=False,

        # Anti-hallucination / anti-repetition guardrails.
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,

        # Optional neutral prompt.
        initial_prompt=initial_prompt,
    )
    out = []
    for s in segments:
        # s.start, s.end, s.text are provided by Faster-Whisper.
        out.append({"start": float(s.start), "end": float(s.end), "text": s.text})
    return out


def text_by_diar_window(diar_segments: List[Dict[str, Any]], asr_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate ASR text by diarization window overlap.

    :param diar_segments: Diarization segments containing ``start``/``end`` boundaries.
    :type diar_segments: List[Dict[str, Any]]
    :param asr_segments: ASR segments with timestamps and transcripts.
    :type asr_segments: List[Dict[str, Any]]
    :returns: List of diarization segments with aggregated ``text`` fields.
    :rtype: List[Dict[str, Any]]
    """
    def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    results = []
    for d in diar_segments:
        d0, d1 = float(d["start"]), float(d["end"])
        buf = []
        for a in asr_segments:
            if overlap(d0, d1, a["start"], a["end"]) > 0.0:
                buf.append(a["text"])
        results.append({**d, "text": " ".join(t for t in buf if t).strip()})
    return results


def transcribe_segments(
    model: Any,
    wav_path: str | Path,
    diar_segments: List[Dict[str, Any]],
    language: str | None = None,
    initial_prompt: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Use Faster-Whisper defaults to transcribe diarised segments.

    :param model: Faster-Whisper compatible model instance.
    :type model: Any
    :param wav_path: Path to the audio file to transcribe.
    :type wav_path: str | Path
    :param diar_segments: Diarization results providing time windows.
    :type diar_segments: List[Dict[str, Any]]
    :param language: Optional forced language code supplied to the decoder.
    :type language: str | None
    :param initial_prompt: Optional context prompt forwarded to Faster-Whisper.
    :type initial_prompt: str | None
    :returns: ASR segment list describing decoded windows.
    :rtype: List[Dict[str, Any]]
    """
    asr_segments = transcribe_full(
        model,
        wav_path,
        language=language,
        initial_prompt=initial_prompt,
    )
    # diar_text = text_by_diar_window(diar_segments, asr_segments)  # Keep disabled; it breaks alignment.
    return asr_segments

# def attach_speakers(
#     diar_segments: List[Dict[str, Any]],
#     asr_segments: List[Dict[str, Any]],
# ) -> List[Dict[str, Any]]:
#     """
#     Assign diarization speaker labels to ASR segments.
#     Policy: attach each ASR segment to the diarization window with the largest overlap.
#     """
#     def overlap(a: Tuple[int,int], b: Tuple[int,int]) -> int:
#         return max(0, min(a[1], b[1]) - max(a[0], b[0]))
#
#     labeled: List[Dict[str, Any]] = []
#     for asr in asr_segments:
#         a_span = (asr["start"], asr["end"])
#         best_spk, best_ov = None, 0
#         for diar in diar_segments:
#             d_span = (diar["start"], diar["end"])
#             ov = overlap(a_span, d_span)
#             if ov > best_ov:
#                 best_ov = ov
#                 best_spk = diar.get("speaker")
#         labeled.append({**asr, "speaker": int(best_spk) if best_spk is not None else -1})
#     return labeled

def attach_speakers(
    diar_segments: List[Dict[str, Any]],
    asr_segments: List[Dict[str, Any]],
    eps: float = 0.05,        # 50 ms tolerance on segment boundaries
    gap_tol: float = 0.30     # 300 ms tolerance when choosing the nearest diar segment
) -> List[Dict[str, Any]]:
    """
    Assign a speaker label to each ASR segment while tolerating minor misalignments.

    If diarization returns a single speaker, the label is propagated across every
    ASR segment. Otherwise, the function searches for the diarization segment that
    contains the ASR midpoint (with a ± ``eps`` margin), falls back to the maximum
    overlap, and finally picks the nearest diarization segment when the gap is below
    ``gap_tol``.

    :param diar_segments: Diarization segments with ``start``/``end`` and speaker labels.
    :type diar_segments: List[Dict[str, Any]]
    :param asr_segments: ASR segments to be labelled with speakers.
    :type asr_segments: List[Dict[str, Any]]
    :param eps: Margin applied when checking whether the midpoint is contained.
    :type eps: float
    :param gap_tol: Maximum allowed distance when falling back to nearest diarization segment.
    :type gap_tol: float
    :returns: ASR segments enriched with a ``speaker`` key.
    :rtype: List[Dict[str, Any]]
    """

    if not diar_segments:
        # No diarization results: assign speaker 0 everywhere.
        return [{**a, "speaker": 0} for a in asr_segments]

    # Normalise types and sort by start time.
    diar = []
    for d in diar_segments:
        ds = float(d["start"])
        de = float(d["end"])
        spk = int(d.get("speaker", 0))
        if de < ds:
            ds, de = de, ds
        diar.append({"start": ds, "end": de, "speaker": spk})
    diar.sort(key=lambda x: x["start"])

    # Trivial case: a single detected speaker across all segments.
    speakers = {d["speaker"] for d in diar}
    if len(speakers) == 1:
        spk0 = int(next(iter(speakers)))
        return [{**a, "speaker": spk0} for a in asr_segments]

    def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    labeled: List[Dict[str, Any]] = []

    for a in asr_segments:
        a0 = float(a["start"])
        a1 = float(a["end"])
        if a1 < a0:
            a0, a1 = a1, a0
        amid = 0.5 * (a0 + a1)

        # 1) Diar segment that contains the midpoint (with ± eps tolerance).
        containing = [d for d in diar if (d["start"] - eps) <= amid <= (d["end"] + eps)]
        if containing:
            # If several segments qualify, keep the one with maximum overlap.
            pick = max(containing, key=lambda d: overlap((a0, a1), (d["start"] - eps, d["end"] + eps)))
            labeled.append({**a, "speaker": int(pick["speaker"])})
            continue

        # 2) No containing segment: pick the maximum overlap (with margins).
        best = None
        best_ov = -1.0
        for d in diar:
            ov = overlap((a0, a1), (d["start"] - eps, d["end"] + eps))
            if ov > best_ov:
                best_ov = ov
                best = d
        if best and best_ov > 0:
            labeled.append({**a, "speaker": int(best["speaker"])})
            continue

        # 3) Fallback: pick the closest segment if the gap is reasonable.
        def dist_to_segment(x0: float, x1: float, y0: float, y1: float) -> float:
            # Minimum distance between [x0, x1] and [y0, y1].
            if x1 < y0:
                return y0 - x1
            if y1 < x0:
                return x0 - y1
            return 0.0  # segments overlap or touch.

        nearest = min(diar, key=lambda d: dist_to_segment(a0, a1, d["start"], d["end"]))
        dmin = dist_to_segment(a0, a1, nearest["start"], nearest["end"])
        if dmin <= gap_tol:
            labeled.append({**a, "speaker": int(nearest["speaker"])})
        else:
            # Ultimate fallback: reuse the most common nearby speaker (first entry).
            labeled.append({**a, "speaker": int(diar[0]["speaker"])})

    return labeled
