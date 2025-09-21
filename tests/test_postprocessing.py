# tests/test_text_export.py
from __future__ import annotations
import io
import pathlib
import sys
from typing import List, Dict

# --- Rendez le paquet 'src' importable si vous lancez pytest depuis la racine du repo ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.postprocessing.text_export import (  # type: ignore
    aggregate_text_per_speaker,
    write_single_block_per_speaker_txt,
    _clean_text,
)

def _mk_seg(start: float, end: float, text: str, speaker) -> Dict:
    return {"start": start, "end": end, "text": text, "speaker": speaker}


# -------------------------
# 1) Nettoyage de texte
# -------------------------
def test_clean_text_basic_punctuation_and_stutter():
    raw = "D'ach-t-t-t-t.  c'est   bien  !   on  va  le  faire ,  d'accord  ? "
    cleaned = _clean_text(raw)
    # Doit supprimer le bégaiement, normaliser espaces/ponctuation, et capitaliser en début de phrase
    assert "D’ach." in cleaned or "D’ach" in cleaned
    assert "C’est bien!" in cleaned
    assert "faire, d’accord?" in cleaned
    # Pas d'espaces multiples
    assert "  " not in cleaned


# -------------------------
# 2) Agrégation mono-locuteur
# -------------------------
def test_aggregate_single_speaker_concatenates_all_in_order():
    segs = [
        _mk_seg(0.0, 1.0, "bonjour à tous", 0),
        _mk_seg(1.1, 2.0, "on commence.", 0),
        _mk_seg(2.2, 2.8, "merci !", 0),
    ]
    blocks = aggregate_text_per_speaker(segs, force_single_speaker=True)
    assert len(blocks) == 1
    spk, text = blocks[0]
    assert spk == 0
    # Tous les morceaux doivent être concaténés dans l'ordre
    assert "Bonjour à tous" in text
    assert "On commence." in text
    assert "Merci!" in text


# -------------------------
# 3) Agrégation multi-locuteur + ordre d'apparition
# -------------------------
def test_aggregate_multiple_speakers_order_by_first_appearance():
    segs = [
        _mk_seg(5.0, 6.0, "deuxième locuteur, premier passage", 1),
        _mk_seg(0.0, 1.0, "premier locuteur, ouverture", 0),
        _mk_seg(1.2, 2.0, "premier locuteur, suite", 0),
        _mk_seg(6.5, 7.0, "deuxième locuteur, suite", 1),
    ]
    blocks = aggregate_text_per_speaker(segs, force_single_speaker=False)
    # 2 blocs attendus
    assert len(blocks) == 2
    # Triés par 1ère apparition -> speaker 0 d'abord
    assert blocks[0][0] == 0
    assert blocks[1][0] == 1
    # Contenu concaténé pour chaque locuteur
    assert "Premier locuteur, ouverture" in blocks[0][1]
    assert "Premier locuteur, suite" in blocks[0][1]
    assert "Deuxième locuteur, premier passage" in blocks[1][1]
    assert "Deuxième locuteur, suite" in blocks[1][1]


# -------------------------
# 4) force_single_speaker fusionne tout
# -------------------------
def test_force_single_speaker_merges_all_speakers():
    segs = [
        _mk_seg(0.0, 0.5, "bonjour", 0),
        _mk_seg(0.6, 1.0, "oui bonjour", 1),
        _mk_seg(1.2, 2.0, "on y va", 0),
    ]
    blocks = aggregate_text_per_speaker(segs, force_single_speaker=True)
    assert len(blocks) == 1
    spk, text = blocks[0]
    assert spk == 0
    assert "Bonjour" in text and "Oui bonjour" in text and "On y va" in text


# -------------------------
# 5) Écriture du .txt (header plain)
# -------------------------
def test_write_single_block_per_speaker_txt_plain(tmp_path):
    segs = [
        _mk_seg(0.0, 0.5, "salut", 0),
        _mk_seg(1.0, 1.5, "ça va", 0),
        _mk_seg(2.0, 2.5, "très bien merci", 1),
    ]
    out_file = tmp_path / "out.txt"
    write_single_block_per_speaker_txt(segs, out_file, force_single_speaker=False, header_style="plain")
    data = out_file.read_text(encoding="utf-8")
    # Doit contenir deux blocs, chacun sur au moins une ligne commençant par SPEAKER_X:
    assert "SPEAKER_1:" in data
    assert "SPEAKER_2:" in data
    # Pas de timestamps attendus
    assert "[" not in data and "]" not in data


# -------------------------
# 6) Écriture du .txt (header titre)
# -------------------------
def test_write_single_block_per_speaker_txt_title_header(tmp_path):
    segs = [
        _mk_seg(0.0, 0.5, "bonjour", 0),
        _mk_seg(0.6, 1.0, "on commence", 0),
    ]
    out_file = tmp_path / "out_title.txt"
    write_single_block_per_speaker_txt(segs, out_file, force_single_speaker=False, header_style="title")
    data = out_file.read_text(encoding="utf-8")
    # Format "titre" = en-tête sur une ligne puis paragraphe
    assert "SPEAKER_1\n" in data
    assert "Bonjour" in data and "On commence" in data


# -------------------------
# 7) Segments vides / bruit
# -------------------------
def test_empty_and_tiny_segments_are_ignored():
    segs = [
        _mk_seg(0.0, 0.1, "", 0),
        _mk_seg(0.1, 0.2, " ", 0),
        _mk_seg(0.2, 1.0, "ok", 0),
    ]
    blocks = aggregate_text_per_speaker(segs)
    assert len(blocks) == 1
    assert blocks[0][1].strip() != "" and "Ok" in blocks[0][1]
