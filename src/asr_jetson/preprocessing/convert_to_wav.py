#!/usr/bin/env python3
"""
Convert assorted audio files (WebM, MP4, M4A, etc.) to 16 kHz mono WAV files.
Compatible with pyannote, Whisper, and the diarization pipeline.
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
    Convert a single audio file into a 16 kHz mono WAV file.

    :param input_path: Source file path (WebM, MP4, and similar formats).
    :type input_path: Union[str, Path]
    :param output_path: Destination path (defaults to swapping the suffix to ``.wav``).
    :type output_path: Optional[Union[str, Path]]
    :param sample_rate: Target sampling rate in Hertz.
    :type sample_rate: int
    :param channels: Target number of channels; ``1`` (mono) is recommended for diarization.
    :type channels: int
    :param use_ffmpeg: Use ``ffmpeg`` when available, otherwise fall back to pure Python libraries.
    :type use_ffmpeg: bool
    :returns: Path to the generated WAV file.
    :rtype: Path
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    # Generate the output path.
    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Method 1: ``ffmpeg`` (fastest and most reliable).
    if use_ffmpeg and shutil.which("ffmpeg"):
        print(f"[INFO] Conversion avec ffmpeg : {input_path.name} → {output_path.name}")
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-ar", str(sample_rate),  # Sample rate
            "-ac", str(channels),  # Mono
            "-sample_fmt", "s16",  # 16-bit PCM
            "-y",  # overwrite if it already exists
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

    # Method 2: pydub (fallback when ffmpeg is unavailable).
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

    # Method 3: soundfile + torchaudio (when installed).
    try:
        import torchaudio
        import soundfile as sf
        print(f"[INFO] Conversion avec torchaudio : {input_path.name} → {output_path.name}")

        waveform, sr = torchaudio.load(str(input_path))

        # Resample when needed.
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        # Convert to mono.
        if waveform.shape[0] > channels:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Persist the waveform.
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
        extensions: tuple[str, ...] = (".webm", ".mp4", ".m4a", ".ogg", ".opus"),
        **kwargs
) -> list[Path]:
    """
    Convert every audio file in a directory.

    :param input_dir: Directory containing the source files.
    :type input_dir: Union[str, Path]
    :param output_dir: Destination directory (defaults to the input directory).
    :type output_dir: Optional[Union[str, Path]]
    :param extensions: Extensions that should be converted.
    :type extensions: tuple[str, ...]
    :param kwargs: Additional parameters forwarded to :func:`convert_to_wav`.
    :type kwargs: dict
    :returns: List of generated WAV files.
    :rtype: list[Path]
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


# === CLI usage ===
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
            # Convert a single file.
            convert_to_wav(
                input_path,
                args.output,
                sample_rate=args.sr,
                channels=channels
            )
        elif input_path.is_dir():
            # Batch conversion.
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
