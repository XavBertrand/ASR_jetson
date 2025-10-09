# src/asr/transcribe.py
from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Iterable, Tuple



def _sec(x_samples: int, sr: int = 16000) -> float:
    return x_samples / float(sr)


# def transcribe_segments(model, wav_path, diar_segments, language=None):
#     """
#     Transcrit en s'appuyant sur les defaults stables de faster-whisper.
#     Fenêtre audio par [start,end] pour réduire la charge côté encode().
#     """
#     results = []
#     for seg in diar_segments:
#         start = float(seg.get("start", 0.0))
#         end = float(seg.get("end", 0.0))
#         if end <= start or (end - start) < 0.05:
#             continue
#
#         try:
#             # Fenêtre : on passe bien la durée ciblée à faster-whisper
#             segments, info = model.transcribe(
#                 wav_path,
#                 language=language,     # None = auto
#                 task="transcribe",
#                 vad_filter=False,      # VAD déjà faite en amont
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

def transcribe_full(model, wav_path, language=None):
    """
    Un seul passage ASR sur tout l'audio.
    Retourne une liste de segments ASR: [{'start': float, 'end': float, 'text': str}, ...]
    """
    segments, info = model.transcribe(
        wav_path,
        language=language,     # None = auto
        task="transcribe",
        vad_filter=False,      # on a déjà fait la VAD en amont
        chunk_length=40,       # garde petit pour la stabilité WSL2 si besoin
        word_timestamps=False,
    )
    out = []
    for s in segments:
        # s.start, s.end, s.text sont fournis par faster-whisper
        out.append({"start": float(s.start), "end": float(s.end), "text": s.text})
    return out


def text_by_diar_window(diar_segments, asr_segments):
    """
    Agrège le texte ASR par fenêtre de diarisation.
    Pour chaque segment diar, concatène le texte des segments ASR qui chevauchent.
    """
    def overlap(a0, a1, b0, b1):
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

def transcribe_segments(model, wav_path, diar_segments, language=None):
    """
    Transcrit en s'appuyant sur les defaults stables de faster-whisper.
    Fenêtre audio par [start,end] pour réduire la charge côté encode().
    """
    asr_segments = transcribe_full(model, wav_path, language=language)
    diar_text = text_by_diar_window(diar_segments, asr_segments)
    return diar_text

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
