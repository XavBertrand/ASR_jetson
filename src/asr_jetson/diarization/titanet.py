import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from pathlib import Path

# # Conserve la compat compat tests (PROJECT_ROOT vient de tests/conftest)
# try:
#     from tests.conftest import PROJECT_ROOT  # type: ignore
# except Exception:
#     PROJECT_ROOT = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _model_search_dirs(env_dir: Optional[str] = None) -> list[Path]:
    """
    Ordre de recherche pour le .nemo :
      1) ASR_NEMO_DIR (si défini)
      2) PROJECT_ROOT/models/nemo
      3) /opt/models/nemo  (utile en Docker Jetson)
    """
    dirs: list[Path] = []
    if env_dir:
        dirs.append(Path(env_dir))
    if PROJECT_ROOT is not None:
        dirs.append(PROJECT_ROOT / "models" / "nemo")
    dirs.append(Path("/opt/models/nemo"))
    # dédoublonne en gardant l’ordre
    seen = set()
    uniq = []
    for d in dirs:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq


def load_titanet(
    device: str = "cpu",
    local_model: Optional[str] = None,
    local_dir: Optional[str] = None,
    filename: str = "titanet_s.nemo",
    allow_downloads: Optional[bool] = None,
    model_name: Optional[str] = None,
):
    """
    Charge un modèle d'embedding speaker (TitaNet-* en priorité).
    Ordre: local (.nemo) -> from_pretrained sur un nom préféré -> détection auto
    -> save local (si possible).

    Env utiles :
      - ASR_NEMO_DIR : dossier pour les .nemo (ex: src/models/nemo ou /opt/models/nemo)
      - ASR_ALLOW_MODEL_DOWNLOADS=0/1 : autoriser les téléchargements (défaut 1)
      - ASR_TITANET_NAME : forcer un nom NeMo (ex: titanet_large)
    """
    # via env
    if allow_downloads is None:
        allow_downloads = os.getenv("ASR_ALLOW_MODEL_DOWNLOADS", "1").lower() in ("1", "true", "yes")
    if model_name is None:
        model_name = os.getenv("ASR_TITANET_NAME")  # facultatif

    # 1) chemins candidats (local)
    candidates: list[Path] = []
    if local_model:
        candidates.append(Path(local_model))
    for d in _model_search_dirs(os.getenv("ASR_NEMO_DIR", local_dir)):
        candidates.append(Path(d) / filename)

    # 2) essai local
    for p in candidates:
        if p.is_file():
            print(f"[INFO] Loading speaker model from local file: {p}")
            model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(str(p), map_location=device)
            model.to(device)
            model.eval()
            return model

    if not allow_downloads:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "Speaker model introuvable localement et téléchargements désactivés "
            "(ASR_ALLOW_MODEL_DOWNLOADS=0). "
            f"Emplacement(s) essayé(s) : {tried}"
        )

    # 3) essai from_pretrained avec un ordre de préférence
    preferred_names: list[str] = []
    if model_name:
        preferred_names.append(model_name)
    # noms les plus courants selon les versions NeMo
    preferred_names += [
        "titanet_s",
        "titanet_small",
        "titanet_large",
        "titanet_large_v1.1",
        "titanet_large_384",
        "titanet_large_512",
        "ecapa_tdnn",            # fallback plausible si pas de TitaNet
        "ecapa_tdnn_small",
    ]

    # on dédupliques en gardant l’ordre
    seen = set()
    ordered = [n for n in preferred_names if not (n in seen or seen.add(n))]

    # essaye directement ces noms
    for name in ordered:
        try:
            print(f"[INFO] Trying NeMo from_pretrained(model_name='{name}')")
            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(name, map_location=device)
            # sauvegarde locale pour les prochains runs
            for d in _model_search_dirs(os.getenv("ASR_NEMO_DIR", local_dir)):
                try:
                    d.mkdir(parents=True, exist_ok=True)
                    out_path = d / filename
                    model.save_to(str(out_path))
                    print(f"[INFO] Saved speaker model to {out_path}")
                    break
                except Exception:
                    continue
            model.to(device)
            model.eval()
            return model
        except Exception:
            continue  # on tente le suivant

    # 4) détection auto : interroge la registry et choisit un nom dispo
    try:
        avail = nemo_asr.models.EncDecSpeakerLabelModel.list_available_models()
        # `avail` peut être liste de str ou d'objets avec .pretrained_model_name
        names = []
        for a in avail:
            if isinstance(a, str):
                names.append(a)
            else:
                # objet de type PretrainedModelInfo
                n = getattr(a, "pretrained_model_name", None) or getattr(a, "name", None)
                if isinstance(n, str):
                    names.append(n)

        # filtre sur 'titanet' d'abord, sinon 'ecapa'
        titanet = [n for n in names if "titanet" in n.lower()]
        ecapa   = [n for n in names if "ecapa" in n.lower()]

        auto_order = titanet + ecapa + names  # essaye titanet puis ecapa puis tout
        # dédupl.
        seen2 = set()
        auto_order = [n for n in auto_order if not (n in seen2 or seen2.add(n))]

        for name in auto_order:
            try:
                print(f"[INFO] Auto-detected NeMo model '{name}', trying from_pretrained...")
                model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(name, map_location=device)
                for d in _model_search_dirs(os.getenv("ASR_NEMO_DIR", local_dir)):
                    try:
                        d.mkdir(parents=True, exist_ok=True)
                        out_path = d / filename
                        model.save_to(str(out_path))
                        print(f"[INFO] Saved speaker model to {out_path}")
                        break
                    except Exception:
                        continue
                model.to(device)
                model.eval()
                return model
            except Exception:
                continue

        # si rien trouvé :
        raise RuntimeError(
            "Aucun modèle compatible (TitaNet/ECAPA) trouvé via list_available_models(). "
            f"Disponibles: {names}"
        )
    except Exception as e:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "Speaker model introuvable localement et échec from_pretrained().\n"
            f"Chemins testés: {tried}\n"
            f"Détails: {e}"
        )



def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalisation L2 par vecteur (N, D) -> (N, D)."""
    denom = torch.clamp(torch.norm(x, p=2, dim=-1, keepdim=True), min=eps)
    return x / denom


def extract_embeddings(
    model,
    wav_path,
    segments: List[Dict],
    device: str = "cpu",
    batch_size: int = 16,
    min_len_samples: int = 1600,  # ~0.1s à 16k
    l2_normalize: bool = True,
) -> np.ndarray:
    """
    Extrait des embeddings TitaNet pour une liste de segments (indices en samples @16k).

    - Lit/convertit l'audio en mono/16k si besoin.
    - Batcher les segments avec padding dynamique et lengths.
    - Normalise L2 les embeddings (optionnel).

    Returns:
        numpy array (num_segments, emb_dim) — peut être vide (0, D).
    """
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    # Mono sûr
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    signal = waveform.to(device)

    # Clamp + padding éventuel des segments très courts
    chunks: List[torch.Tensor] = []
    lengths: List[int] = []

    sig_len = int(signal.shape[1])
    for seg in segments:
        start, end = int(seg["start"]), int(seg["end"])
        start = max(0, min(start, sig_len))
        end = max(0, min(end, sig_len))

        if end <= start:
            continue

        chunk = signal[:, start:end]
        if chunk.shape[1] < min_len_samples:
            pad = min_len_samples - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        # TitaNet attend [B, T], on s'assure d'un batch plus tard
        # Ici chunk est [1, T], c'est parfait
        chunks.append(chunk.squeeze(0))  # (T,)
        lengths.append(int(chunk.shape[1]))

    if not chunks:
        # Taille d'embedding connue de TitaNet-S : 192
        return np.zeros((0, 192), dtype=np.float32)

    # Batching avec padding max_len par batch
    embeddings_all: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(chunks), batch_size):
            batch_sig = chunks[i : i + batch_size]
            batch_len = lengths[i : i + batch_size]

            max_len = max(batch_len)
            # Empile en [B, T] avec padding right
            padded = torch.stack(
                [torch.nn.functional.pad(x, (0, max_len - x.shape[0])) for x in batch_sig],
                dim=0,
            ).to(device)  # (B, T)

            len_tensor = torch.tensor(batch_len, device=device, dtype=torch.int64)

            # forward -> tuple(embeddings, ...) parfois ; on récupère le 1er
            out = model.forward(input_signal=padded, input_signal_length=len_tensor)
            if isinstance(out, tuple):
                out = out[0]

            # out attendu [B, D]
            if out.ndim == 1:
                out = out.unsqueeze(0)

            if l2_normalize:
                out = _l2_normalize(out)

            embeddings_all.append(out.detach().cpu().to(torch.float32).numpy())

    embs = np.vstack(embeddings_all) if embeddings_all else np.zeros((0, 192), dtype=np.float32)
    return embs
