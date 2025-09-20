# src/asr/transcribe.py
from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Iterable, Tuple

def _sec(x_samples: int, sr: int = 16000) -> float:
    return x_samples / float(sr)

def transcribe_segments(
    model,
    audio_path: str | Path,
    vad_or_diar_segments: List[Dict[str, int]],
    language: str | None = None,
    beam_size: int = 1,
    no_speech_threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Transcrit uniquement les fenêtres [start:end] (échantillons 16 kHz) pour accélérer.
    Retourne une liste de segments:
      {start, end, start_s, end_s, text, words?}
    """
    from faster_whisper import decode_audio

    audio_path = str(Path(audio_path))
    # On charge *une fois* le waveform (mono, 16k recommandé en amont de ta pipeline)
    samples = decode_audio(audio_path)  # float32, 16000 Hz si fichier déjà 16k (sinon faster-whisper resample)

    out: List[Dict[str, Any]] = []
    for _, seg in enumerate(tqdm(vad_or_diar_segments)):
        s, e = int(seg["start"]), int(seg["end"])
        if e <= s:
            continue
        chunk = samples[s:e]  # sous-signal

        # Transcription chunkée (note: faster-whisper accepte des arrays numpy)
        segments, _info = model.transcribe(
            chunk,
            language=language,
            beam_size=beam_size,
            vad_filter=False,      # VAD déjà fait
            no_speech_threshold=no_speech_threshold,
            word_timestamps=True,  # pratique pour aligner les mots si voulu
        )

        # Concatène le texte de toutes les sous-parties du chunk
        text_parts = []
        words = []
        for sub in segments:
            if sub.text:
                text_parts.append(sub.text.strip())
            if sub.words:
                # Ajuste les timestamps mots en relatif au chunk
                for w in sub.words:
                    words.append({
                        "word": w.word,
                        "start": s + int(round(w.start * 16000)),
                        "end":   s + int(round(w.end * 16000)),
                        "start_s": _sec(s) + float(w.start),
                        "end_s":   _sec(s) + float(w.end),
                    })

        out.append({
            "start": s, "end": e,
            "start_s": _sec(s), "end_s": _sec(e),
            "text": " ".join(text_parts).strip(),
            "words": words,
        })

    return out

def attach_speakers(
    diar_segments: List[Dict[str, Any]],
    asr_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Associe les labels speaker (issus de ta diarisation) aux segments ASR.
    Politique simple : affecter chaque segment ASR au speaker du segment de diar le plus chevauchant.
    """
    def overlap(a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    labeled: List[Dict[str, Any]] = []
    for asr in asr_segments:
        a_span = (asr["start"], asr["end"])
        best_spk, best_ov = None, 0
        for diar in diar_segments:
            d_span = (diar["start"], diar["end"])
            ov = overlap(a_span, d_span)
            if ov > best_ov:
                best_ov = ov
                best_spk = diar.get("speaker")
        labeled.append({**asr, "speaker": int(best_spk) if best_spk is not None else -1})
    return labeled
