import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torchaudio
import nemo.collections.asr as nemo_asr

# Conserve la compat compat tests (PROJECT_ROOT vient de tests/conftest)
try:
    from tests.conftest import PROJECT_ROOT  # type: ignore
except Exception:
    PROJECT_ROOT = None


def load_titanet(
    device: str = "cpu",
    local_model: Optional[str] = None,
):
    """
    Charge TitaNet-S pour l'extraction d'embeddings orateurs.

    Args:
        device: "cpu" ou "cuda[:id]"
        local_model: chemin d'un .nemo local (tests), sinon tente NGC.

    Returns:
        EncDecSpeakerLabelModel (en mode eval, sur le bon device)
    """
    if local_model is None and PROJECT_ROOT is not None:
        local_model = os.path.join(PROJECT_ROOT, "models", "nemo", "titanet-s.nemo")

    if local_model and os.path.exists(local_model):
        print(f"[INFO] Loading TitaNet from local file: {local_model}")
        model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            local_model, map_location=device
        )
    else:
        try:
            print("[INFO] Loading TitaNet-S from NGC")
            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                "titanet_s", map_location=device
            )
        except Exception as e:
            raise FileNotFoundError(
                "TitaNet-S not found locally or on NGC.\n"
                f"Tried local_model={local_model}\n"
                f"Error: {e}"
            )

    model.to(device)
    model.eval()  # IMPORTANT pour BN / Dropout
    return model


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
