import torch
import torchaudio
import numpy as np
import nemo.collections.asr as nemo_asr
from typing import List, Dict
import os

import nemo.collections.asr as nemo_asr

from tests.conftest import PROJECT_ROOT


def load_titanet(device="cpu", local_model=os.path.join(PROJECT_ROOT, "models", "nemo", "titanet-s.nemo")):
    """
    Load TitaNet-S model for speaker embedding extraction.

    Args:
        device (str): "cpu" or "cuda"
        local_model (str): path to a local .nemo file for tests

    Returns:
        EncDecSpeakerLabelModel
    """
    # Si un modèle local existe → priorité
    if os.path.exists(local_model):
        print(f"[INFO] Loading TitaNet from local file: {local_model}")
        model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(local_model, map_location=device)
    else:
        # Sinon essayer via NGC
        try:
            print("[INFO] Loading TitaNet-S from NGC")
            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("titanet_s", map_location=device)
        except Exception as e:
            raise FileNotFoundError(
                f"TitaNet-S not found locally or on NGC.\n"
                f"Tried local_model={local_model}\n"
                f"Error: {e}"
            )

    # IMPORTANT: Mettre le modèle en mode evaluation pour l'inférence
    model.eval()

    return model


def extract_embeddings(model, wav_path, segments, device="cpu"):
    waveform, sr = torchaudio.load(wav_path)

    # Resample si nécessaire
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    signal = waveform.to(device)

    embeddings = []

    # S'assurer que le modèle est en mode evaluation
    model.eval()

    with torch.no_grad():  # Désactiver les gradients pour l'inférence
        for seg in segments:
            start, end = seg["start"], seg["end"]

            # Clamp indices pour éviter dépassement
            start = max(0, min(start, signal.shape[1]))
            end = max(0, min(end, signal.shape[1]))

            chunk = signal[:, start:end]
            if chunk.shape[1] == 0:
                continue

            # Vérifier la longueur minimale du segment
            min_length = 1600  # 0.1 seconde à 16kHz
            if chunk.shape[1] < min_length:
                # Padding si le segment est trop court
                pad_length = min_length - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_length))

            if chunk.ndim > 2:
                chunk = chunk.squeeze()  # devient [samples]
            if chunk.ndim == 1:
                chunk = chunk.unsqueeze(0)  # devient [1, samples]

            emb = model.forward(
                input_signal=chunk,
                input_signal_length=torch.tensor([chunk.shape[1]]).to(device),
            )

            # model.forward() retourne un tuple, récupérer les embeddings
            if isinstance(emb, tuple):
                # Debug: voir le contenu du tuple si nécessaire
                # print(f"[DEBUG] model.forward() returned tuple of length {len(emb)}")
                # print(f"[DEBUG] Types: {[type(x) for x in emb]}")
                emb = emb[0]  # Premier élément du tuple contient les embeddings

            # Vérifier que emb est bien un tensor
            if not isinstance(emb, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(emb)}")

            embeddings.append(emb.cpu().detach().numpy())

    if len(embeddings) == 0:
        return np.zeros((0, model._cfg.get("emb_dim", 192)))

    return np.vstack(embeddings)