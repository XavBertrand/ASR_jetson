# tests/test_txt_file_format.py
from __future__ import annotations
import re
import pathlib
import sys

# Rendre 'src' importable si pytest est lancé depuis la racine
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from postprocessing.text_export import write_single_block_per_speaker_txt  # type: ignore


def _mk_seg(start: float, end: float, text: str, speaker):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def test_txt_plain_format_rules(tmp_path):
    """Le fichier .txt en mode 'plain' doit :
    - avoir un seul bloc par locuteur (une ligne d'en-tête + un paragraphe),
    - contenir exactement une ligne vide entre blocs,
    - ne pas contenir de timestamps ni de doubles espaces,
    - ne pas avoir d'espace avant la ponctuation , . ; : ! ?,
    - se terminer par un '\n' final.
    """
    segs = [
        _mk_seg(0.0, 1.0, "bonjour à tous", 0),
        _mk_seg(1.1, 2.0, "on commence.", 0),
        _mk_seg(2.1, 3.0, "très bien, merci", 1),
        _mk_seg(3.1, 4.0, "parfait ! on continue", 1),
    ]
    out_file = tmp_path / "plain.txt"
    write_single_block_per_speaker_txt(
        segs, out_file, force_single_speaker=False, header_style="plain"
    )

    data = out_file.read_text(encoding="utf-8")

    # 0) Fichier se termine par un saut de ligne
    assert data.endswith("\n")

    # 1) En-têtes attendus exactement une fois chacun
    headers = re.findall(r"(?m)^SPEAKER_\d+:", data)
    assert headers == ["SPEAKER_1:", "SPEAKER_2:"], f"Headers found: {headers}"

    # 2) Extraire les blocs: "SPEAKER_X: <paragraphe>" séparés par UNE ligne vide
    blocks = re.findall(
        r"(?ms)^SPEAKER_(\d+): (.*?)(?:\n\n|\Z)", data
    )
    assert len(blocks) == 2

    # 3) Chaque bloc contient les phrases concaténées du locuteur correspondant
    spk1_num, spk1_text = blocks[0]
    spk2_num, spk2_text = blocks[1]
    assert spk1_num == "1" and spk2_num == "2"
    assert "Bonjour à tous" in spk1_text
    assert "On commence." in spk1_text
    assert "Très bien, merci" in spk2_text
    assert "Parfait! On continue" in spk2_text or "Parfait ! On continue" in spk2_text

    # 4) Pas de timestamps, pas de doubles espaces
    assert "[" not in data and "]" not in data
    assert "  " not in data

    # 5) Pas d'espace avant la ponctuation , . ; : ! ?
    assert " ," not in data and " ." not in data and " ;" not in data \
           and " :" not in data and " !" not in data and " ?" not in data

    # 6) Une SEULE ligne vide entre blocs (grâce au join + lignes terminées par \n)
    # i.e., motif "\n\n" présent, mais pas "\n\n\n"
    assert "\n\n" in data and "\n\n\n" not in data


def test_txt_title_format_rules(tmp_path):
    """Le fichier .txt en mode 'title' doit :
    - avoir un en-tête seul sur une ligne (SPEAKER_X) suivi du paragraphe,
    - respecter les mêmes règles de propreté (pas de timestamps, etc.),
    - avoir exactement une ligne vide entre blocs,
    - se terminer par un '\n' final.
    """
    segs = [
        _mk_seg(0.0, 0.8, "bonjour", 0),
        _mk_seg(0.9, 1.5, "on démarre", 0),
        _mk_seg(1.6, 2.4, "ok, bien reçu", 1),
    ]
    out_file = tmp_path / "title.txt"
    write_single_block_per_speaker_txt(
        segs, out_file, force_single_speaker=False, header_style="title"
    )

    data = out_file.read_text(encoding="utf-8")
    assert data.endswith("\n")

    # 1) En-têtes "titre" (pas de ":" sur la même ligne)
    title_headers = re.findall(r"(?m)^SPEAKER_\d+\s*$", data)
    assert title_headers == ["SPEAKER_1", "SPEAKER_2"]

    # 2) Blocs: "SPEAKER_X\n<paragraphe>" séparés par UNE ligne vide
    blocks = re.findall(
        r"(?ms)^SPEAKER_(\d+)\s*\n(.*?)(?:\n\n|\Z)", data
    )
    assert len(blocks) == 2

    spk1_num, spk1_text = blocks[0]
    spk2_num, spk2_text = blocks[1]
    assert spk1_num == "1" and spk2_num == "2"
    assert "Bonjour" in spk1_text and "On démarre" in spk1_text
    assert "Ok, bien reçu" in spk2_text or "OK, bien reçu" in spk2_text

    # 3) Propreté
    assert "[" not in data and "]" not in data
    assert "  " not in data
    assert " ," not in data and " ." not in data and " ;" not in data \
           and " :" not in data and " !" not in data and " ?" not in data

    # 4) Une seule ligne vide entre blocs
    assert "\n\n" in data and "\n\n\n" not in data
