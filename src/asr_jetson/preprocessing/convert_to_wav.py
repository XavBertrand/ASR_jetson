#!/usr/bin/env python3
"""
Convertit des fichiers audio (WebM, MP4, M4A, etc.) en WAV mono 16kHz
Compatible avec pyannote, whisper, et ton pipeline de diarization.
"""

from pathlib import Path
import subprocess
import sys
from typing import Optional, Union
import shutil


def convert_to_wav(
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        use_ffmpeg: bool = True,
) -> Path:
    """
    Convertit un fichier audio en WAV mono 16kHz.

    Args:
        input_path: Chemin du fichier source (WebM, MP4, etc.)
        output_path: Chemin de sortie (défaut: même nom avec .wav)
        sample_rate: Fréquence d'échantillonnage (16000 par défaut)
        channels: Nombre de canaux (1=mono, recommandé pour la diarization)
        use_ffmpeg: Utilise ffmpeg si dispo, sinon pydub/soundfile

    Returns:
        Path du fichier WAV créé
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    # Génère le nom de sortie
    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Méthode 1 : ffmpeg (le plus fiable et rapide)
    if use_ffmpeg and shutil.which("ffmpeg"):
        print(f"[INFO] Conversion avec ffmpeg : {input_path.name} → {output_path.name}")
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-ar", str(sample_rate),  # Sample rate
            "-ac", str(channels),  # Mono
            "-sample_fmt", "s16",  # 16-bit PCM
            "-y",  # Écrase si existe
            str(output_path)
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"[OK] Fichier créé : {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[ERREUR] ffmpeg a échoué : {e.stderr}")
            raise

    # Méthode 2 : pydub (fallback si pas ffmpeg)
    try:
        from pydub import AudioSegment
        print(f"[INFO] Conversion avec pydub : {input_path.name} → {output_path.name}")

        audio = AudioSegment.from_file(str(input_path))
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_channels(channels)
        audio = audio.set_sample_width(2)  # 16-bit

        audio.export(str(output_path), format="wav")
        print(f"[OK] Fichier créé : {output_path}")
        return output_path
    except ImportError:
        pass

    # Méthode 3 : soundfile + torchaudio (si installé)
    try:
        import torchaudio
        import soundfile as sf
        print(f"[INFO] Conversion avec torchaudio : {input_path.name} → {output_path.name}")

        waveform, sr = torchaudio.load(str(input_path))

        # Resample si nécessaire
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        # Convertit en mono
        if waveform.shape[0] > channels:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Sauvegarde
        torchaudio.save(
            str(output_path),
            waveform,
            sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )
        print(f"[OK] Fichier créé : {output_path}")
        return output_path
    except ImportError:
        pass

    raise RuntimeError(
        "Aucune méthode de conversion disponible.\n"
        "Installe ffmpeg (recommandé) ou pydub :\n"
        "  - Ubuntu/Debian : sudo apt install ffmpeg\n"
        "  - macOS : brew install ffmpeg\n"
        "  - Windows : https://ffmpeg.org/download.html\n"
        "  - Ou pip install pydub"
    )


def convert_batch(
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        extensions: tuple = (".webm", ".mp4", ".m4a", ".ogg", ".opus"),
        **kwargs
) -> list[Path]:
    """
    Convertit tous les fichiers audio d'un dossier.

    Args:
        input_dir: Dossier contenant les fichiers sources
        output_dir: Dossier de sortie (défaut: même dossier)
        extensions: Extensions à convertir
        **kwargs: Arguments passés à convert_to_wav()

    Returns:
        Liste des fichiers WAV créés
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*{ext}"))

    if not files:
        print(f"[WARN] Aucun fichier trouvé avec les extensions {extensions}")
        return []

    print(f"[INFO] {len(files)} fichier(s) à convertir")

    converted = []
    for f in files:
        try:
            out_path = output_dir / f.with_suffix(".wav").name
            result = convert_to_wav(f, out_path, **kwargs)
            converted.append(result)
        except Exception as e:
            print(f"[ERREUR] Échec pour {f.name} : {e}")

    print(f"[OK] {len(converted)}/{len(files)} fichiers convertis")
    return converted


# === Utilisation CLI ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convertit des fichiers audio en WAV mono 16kHz"
    )
    parser.add_argument("input", help="Fichier ou dossier à convertir")
    parser.add_argument("-o", "--output", help="Fichier/dossier de sortie")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (défaut: 16000)")
    parser.add_argument("--stereo", action="store_true", help="Garde le stéréo (défaut: mono)")

    args = parser.parse_args()

    input_path = Path(args.input)
    channels = 2 if args.stereo else 1

    try:
        if input_path.is_file():
            # Conversion d'un fichier unique
            convert_to_wav(
                input_path,
                args.output,
                sample_rate=args.sr,
                channels=channels
            )
        elif input_path.is_dir():
            # Conversion batch
            convert_batch(
                input_path,
                args.output,
                sample_rate=args.sr,
                channels=channels
            )
        else:
            print(f"[ERREUR] Chemin invalide : {input_path}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERREUR] {e}")
        sys.exit(1)