# src/postprocessing/text_export.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Tuple

# --- utilitaires légers de nettoyage ---
def _normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s

def _fix_fr_punct(s: str) -> str:
    # Normalise les espaces de base
    s = re.sub(r"\s+", " ", s.strip())

    # 1) Supprimer les espaces AVANT la ponctuation
    #    -> "locuteur , ouverture"  => "locuteur, ouverture"
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)

    # 2) Forcer exactement UN espace APRÈS la ponctuation (si un caractère suit)
    #    -> "bien!on" => "bien! on"
    s = re.sub(r"([,;:!?])(?=\S)", r"\1 ", s)
    s = re.sub(r"([.])(?=\S)", r"\1 ", s)

    # 3) Apostrophes françaises
    s = s.replace(" '", " ’").replace("' ", "’ ").replace("'", "’")
    s = re.sub(r"\b([cdjlmnstCDJLMNST])\s*’\s*([aeéèêiouhAEÉÈÊIOUH])", r"\1’\2", s)

    # 4) Capitalisation simple en début de phrase
    s = re.sub(r"(^|[.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), s)

    # 5) Nettoyage final
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_text(s: str) -> str:
    s = _normalize_ws(s)
    # supprime bégaiements évidents: "re-re-regarde", "t-t-t-..."
    s = re.sub(r"\b(\w{1,4})(?:-\1){1,}\b", r"\1", s, flags=re.IGNORECASE)
    s = _fix_fr_punct(s)
    return s

def _speaker_idx(spk) -> int:
    if isinstance(spk, int):
        return spk
    s = str(spk).upper().replace("SPEAKER_", "").replace("SPK", "").strip()
    try:
        return int(s)
    except Exception:
        return 0

# --- fusion "tout par locuteur" ---
def aggregate_text_per_speaker(
    segments: List[Dict],
    force_single_speaker: bool = False,
    sort_by_first_appearance: bool = True,
    min_segment_chars: int = 1,
) -> List[Tuple[int, str]]:
    """
    Regroupe TOUT le texte de chaque locuteur en UN SEUL BLOC par locuteur.
    Retourne une liste [(speaker_index, texte_concaténé), ...] triée
    par ordre d'apparition du locuteur (ou par index).
    """
    # 1) normaliser et filtrer
    norm = []
    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if len(txt) < min_segment_chars:
            continue
        start = float(seg.get("start", seg.get("start_s", 0.0)) or 0.0)
        spk = 0 if force_single_speaker else _speaker_idx(seg.get("speaker", 0))
        norm.append((start, spk, _clean_text(txt)))

    if not norm:
        return []

    # 2) trier par temps afin de respecter l'ordre d'apparition
    norm.sort(key=lambda t: t[0])

    # 3) cumuler par speaker
    all_text_by_spk = {}
    first_appearance = {}
    for i, (start, spk, txt) in enumerate(norm):
        if spk not in all_text_by_spk:
            all_text_by_spk[spk] = []
            first_appearance[spk] = start
        all_text_by_spk[spk].append(txt)

    # 4) construire un seul paragraphe par speaker
    results = []
    for spk, chunks in all_text_by_spk.items():
        paragraph = _clean_text(" ".join(chunks))
        results.append((spk, paragraph))

    # 5) tri final
    if sort_by_first_appearance:
        results.sort(key=lambda kv: first_appearance.get(kv[0], 0.0))
    else:
        results.sort(key=lambda kv: kv[0])

    return results

def write_single_block_per_speaker_txt(
    segments: List[Dict],
    out_path: Path,
    force_single_speaker: bool = False,
    header_style: str = "plain",  # "plain" -> SPEAKER_1:  ; "title" -> "SPEAKER_1\n"
) -> None:
    """
    Écrit un .txt avec UN SEUL BLOC par locuteur, sans timestamps.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = aggregate_text_per_speaker(
        segments, force_single_speaker=force_single_speaker
    )

    lines = []
    for spk, para in blocks:
        tag = f"SPEAKER_{spk+1}"
        if header_style == "title":
            lines.append(f"{tag}\n{para}\n")
        else:
            lines.append(f"{tag}: {para}\n")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

def write_dialogue_txt(segments, path, one_based=True):
    """
    Écrit:
    SPEAKER_1 : bonjour !!
    SPEAKER_2 : Bonjour, comment vas-tu ?
    ...
    """
    segments_sorted = sorted(
        segments,
        key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0)))
    )
    lines = []
    for seg in segments_sorted:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = int(seg.get("speaker", 0))
        if one_based:
            spk += 1  # SPEAKER_1, SPEAKER_2, ... (au lieu de 0/1)
        lines.append(f"SPEAKER_{spk} : {text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
