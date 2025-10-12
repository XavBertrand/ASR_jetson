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
        chunk_length=30,       # garde petit pour la stabilité WSL2 si besoin
        word_timestamps=False,
        beam_size=5,  # beam search (meilleure ponctuation/capitales)
        temperature=[0.0, 0.2, 0.4],  # beam = 0.0 => déterministe
        condition_on_previous_text=True,  # conserve le contexte entre chunks
        compression_ratio_threshold=2.4,  # garde-fous texte dégénéré
        initial_prompt="Ponctue correctement en français (., ; : ! ?), garde les nombres et noms propres."
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
    # diar_text = text_by_diar_window(diar_segments, asr_segments)  # Ne pas mettre ! Ca fout le bazar
    return asr_segments

# def attach_speakers(
#     diar_segments: List[Dict[str, Any]],
#     asr_segments: List[Dict[str, Any]],
# ) -> List[Dict[str, Any]]:
#     """
#     Associe les labels speaker (issus de ta diarisation) aux segments ASR.
#     Politique simple : affecter chaque segment ASR au speaker du segment de diar le plus chevauchant.
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
    eps: float = 0.05,        # 50 ms de tolérance sur les bords
    gap_tol: float = 0.30     # 300 ms de tolérance pour prendre le plus proche
) -> List[Dict[str, Any]]:
    """
    Associe un speaker à chaque segment ASR en étant tolérant aux minuscules désalignements.
    - Si un seul speaker est présent dans la diarisation, on l'applique à tous les segments ASR.
    - Sinon : on cherche d'abord un diar qui 'contient' le milieu du segment ASR (±eps),
      puis on prend celui qui maximise le chevauchement (avec marges eps),
      sinon on prend le diar le plus proche si l'écart est < gap_tol.
    """

    if not diar_segments:
        # Pas de diar : tout sur speaker 0
        return [{**a, "speaker": 0} for a in asr_segments]

    # Normalise types et trie
    diar = []
    for d in diar_segments:
        ds = float(d["start"])
        de = float(d["end"])
        spk = int(d.get("speaker", 0))
        if de < ds:
            ds, de = de, ds
        diar.append({"start": ds, "end": de, "speaker": spk})
    diar.sort(key=lambda x: x["start"])

    # Cas trivial : un seul speaker détecté -> on l’assigne partout
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

        # 1) Diar "contenant" le milieu (avec tolérance eps)
        containing = [d for d in diar if (d["start"] - eps) <= amid <= (d["end"] + eps)]
        if containing:
            # S'il y en a plusieurs, on prend celui qui a le plus grand chevauchement
            pick = max(containing, key=lambda d: overlap((a0, a1), (d["start"] - eps, d["end"] + eps)))
            labeled.append({**a, "speaker": int(pick["speaker"])})
            continue

        # 2) Pas de contenant : prend max overlap (avec marges eps)
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

        # 3) Fallback : le plus proche si l'écart est raisonnable (gap_tol)
        def dist_to_segment(x0: float, x1: float, y0: float, y1: float) -> float:
            # distance minimale entre [x0,x1] et [y0,y1]
            if x1 < y0:
                return y0 - x1
            if y1 < x0:
                return x0 - y1
            return 0.0  # ils se chevauchent (ou se touchent)

        nearest = min(diar, key=lambda d: dist_to_segment(a0, a1, d["start"], d["end"]))
        dmin = dist_to_segment(a0, a1, nearest["start"], nearest["end"])
        if dmin <= gap_tol:
            labeled.append({**a, "speaker": int(nearest["speaker"])})
        else:
            # Ultime fallback : le speaker majoritaire autour (prend le premier)
            labeled.append({**a, "speaker": int(diar[0]["speaker"])})

    return labeled