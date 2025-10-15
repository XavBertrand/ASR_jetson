import os
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from pathlib import Path

try:
    from speechbrain.inference import EncoderClassifier
    _HAS_SPEECHBRAIN = True
except Exception:
    _HAS_SPEECHBRAIN = False

# # Conserve la compat compat tests (PROJECT_ROOT vient de tests/conftest)
# try:
#     from tests.conftest import PROJECT_ROOT  # type: ignore
# except Exception:
#     PROJECT_ROOT = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class ECAPAWrapper(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )

    @torch.inference_mode()
    def forward(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor):
        # input_signal: (B, T) mono @16k
        # SpeechBrain attend (B, T), retourne (B, 192)
        # encode_batch normalise en interne; on reste en fp32
        emb = self.classifier.encode_batch(input_signal)
        # encode_batch -> (B, 1, D), squeeze le canal
        emb = emb.squeeze(1)  # (B, D)
        return emb

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
    filename: str = "titanet_large.nemo",
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
    # x = x - x.mean(dim=0, keepdim=True)
    denom = torch.clamp(torch.norm(x, p=2, dim=-1, keepdim=True), min=eps)
    return x / denom


def extract_embeddings(
    model,
    wav_path,
    segments: List[Dict],
    device: str = "cpu",
    batch_size: int = 16,
    min_len_samples: int = 8000,  # ~0.5s à 16k, minimum for titanet-S
    l2_normalize: bool = True,
    win_sec: Optional[float] = 1.0,
    hop_sec: Optional[float] = 0.25,
    return_times: bool = False,        # <--- AJOUT
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    signal = waveform.to(device)

    chunks: List[torch.Tensor] = []
    lengths: List[int] = []
    starts_samples: List[int] = []      # <--- AJOUT

    sig_len = int(signal.shape[1])
    use_sliding = (win_sec is not None) and (hop_sec is not None) and (win_sec > 0) and (hop_sec > 0)
    if use_sliding:
        win = int(win_sec * 16000)
        hop = int(hop_sec * 16000)

    ABSOLUTE_MIN_SEC = 0.5  # 0.5s minimum
    segments = [s for s in segments if (float(s["end"]) - float(s["start"])) >= ABSOLUTE_MIN_SEC]

    for seg in segments:
        # ✅ Conversion secondes -> samples
        start = int(float(seg["start"]) * 16000)
        end = int(float(seg["end"]) * 16000)

        start = max(0, min(start, sig_len))
        end = max(0, min(end, sig_len))

        # ✅ Vérification longueur en samples
        if (end - start) < 8000:  # < 0.5s
            continue

        if end <= start:
            continue

        if not use_sliding:
            chunk = signal[:, start:end]
            actual_len = chunk.shape[1]  # ✅ Longueur réelle AVANT padding
            if actual_len < min_len_samples:
                pad = min_len_samples - actual_len
                chunk = torch.nn.functional.pad(chunk, (0, pad))
            chunks.append(chunk.squeeze(0))
            lengths.append(actual_len)  # ✅ Longueur réelle du signal
            # centre de la fenêtre unique
            starts_samples.append(start + chunk.shape[1] // 2)  # <--- AJOUT
        else:
            seg_len = end - start
            if seg_len <= 0:
                continue
            if seg_len < max(min_len_samples, win):
                chunk = signal[:, start:end]
                actual_len = chunk.shape[1]  # ✅ Longueur réelle
                if actual_len < min_len_samples:
                    pad = min_len_samples - actual_len
                    chunk = torch.nn.functional.pad(chunk, (0, pad))
                chunks.append(chunk.squeeze(0))
                lengths.append(actual_len)  # ✅ Longueur réelle
                starts_samples.append(start + chunk.shape[1] // 2)  # <--- AJOUT
                continue

            sub_starts = list(range(start, end - win + 1, hop))
            if not sub_starts or sub_starts[-1] + win < end:
                sub_starts.append(max(start, end - win))

            for s0 in sub_starts:
                s1 = min(s0 + win, end)
                chunk = signal[:, s0:s1]
                actual_len = chunk.shape[1]  # ✅ Longueur réelle
                if actual_len < min_len_samples:
                    pad = min_len_samples - actual_len
                    chunk = torch.nn.functional.pad(chunk, (0, pad))
                chunks.append(chunk.squeeze(0))
                lengths.append(actual_len)  # ✅ Longueur réelle
                starts_samples.append(s0 + chunk.shape[1] // 2)   # <--- AJOUT

    if not chunks:
        return (np.zeros((0, 192), dtype=np.float32), np.zeros((0,), dtype=np.float32)) if return_times \
               else np.zeros((0, 192), dtype=np.float32)

    # ✅ Validation finale : retire les chunks invalides
    valid_indices = [i for i, (c, l) in enumerate(zip(chunks, lengths)) if l >= 8000]

    if not valid_indices:
        if return_times:
            return np.zeros((0, 192), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.zeros((0, 192), dtype=np.float32)

    chunks = [chunks[i] for i in valid_indices]
    lengths = [lengths[i] for i in valid_indices]
    starts_samples = [starts_samples[i] for i in valid_indices]

    embeddings_all: List[np.ndarray] = []
    model.eval()

    ABSOLUTE_MIN = 8000

    with torch.inference_mode():
        for i in range(0, len(chunks), batch_size):
            batch_sig = chunks[i : i + batch_size]
            batch_len = lengths[i : i + batch_size]

            # ✅ Garantit que max_len >= ABSOLUTE_MIN
            max_len = max(max(batch_len), ABSOLUTE_MIN)

            padded = torch.stack(
                [torch.nn.functional.pad(x, (0, max_len - x.shape[0])) for x in batch_sig],
                dim=0,
            ).to(device)

            len_tensor = torch.tensor(batch_len, device=device, dtype=torch.int64)

            # ✅ DEBUG : Vérifier le signal d'entrée
            if i == 0:
                print(f"\n[DEBUG BATCH INPUT]")
                print(f"  Padded shape: {padded.shape}")
                print(f"  Lengths: {batch_len[:5]}")
                print(f"  Signal stats: min={padded.min():.4f}, max={padded.max():.4f}, mean={padded.mean():.4f}")
                print(f"  Signal std: {padded.std():.4f}")

            logits, out = model.forward(input_signal=padded, input_signal_length=len_tensor)
            # ✅ TitaNet retourne (logits, embeddings) ou juste embeddings
            # if isinstance(out, tuple):
            #     # NeMo retourne souvent (logits, embeddings) ou (embeddings, lengths)
            #     # Les embeddings sont généralement le dernier élément de taill/e (batch, 192)
            #     candidates = [o for o in out if isinstance(o, torch.Tensor)]
            #
            #     # Trouve le tensor avec dimension ~192
            #     for candidate in candidates:
            #         if candidate.ndim == 2 and candidate.shape[1] in [192, 256, 512]:
            #             out = candidate
            #             break
            #     else:
            #         # Fallback : prend le premier tensor 2D
            #         out = out[-1] if len(out) > 1 else out[0]
            #
            # if out.ndim == 1:
            #     out = out.unsqueeze(0)

            # ✅ Vérification de dimension
            if out.shape[1] > 1000:  # Audio brut au lieu d'embeddings !
                raise RuntimeError(
                    f"Embeddings invalides: shape {out.shape}. "
                    f"Attendu: (batch, 192-512), obtenu: (batch, {out.shape[1]}). "
                    f"Le modèle TitaNet ne retourne pas les embeddings."
                )
            if out.ndim == 1:
                out = out.unsqueeze(0)

            # ✅ DEBUG : Embeddings AVANT normalisation
            if i == 0:
                print(f"\n[DEBUG EMBEDDINGS RAW]")
                print(f"  Shape: {out.shape}")
                print(f"  Stats: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
                print(f"  Std: {out.std():.4f}")
                print(f"  Norme L2 moyenne: {torch.norm(out, dim=1).mean():.4f}")

                # Vérifier si tous les embeddings sont identiques
                if len(out) > 1:
                    diff = torch.abs(out[0] - out[1]).sum()
                    print(f"  Différence entre emb[0] et emb[1]: {diff:.6f}")

            if l2_normalize:
                out = _l2_normalize(out)

            # ✅ DEBUG : Embeddings APRÈS normalisation
            if i == 0:
                print(f"\n[DEBUG EMBEDDINGS NORMALIZED]")
                print(f"  Norme L2 moyenne: {torch.norm(out, dim=1).mean():.4f}")
                if len(out) > 1:
                    cos_sim = torch.nn.functional.cosine_similarity(out[0:1], out[1:2])
                    print(f"  Similarité cosine entre emb[0] et emb[1]: {cos_sim.item():.4f}")

            embeddings_all.append(out.detach().cpu().to(torch.float32).numpy())

    embs = np.vstack(embeddings_all) if embeddings_all else np.zeros((0, 192), dtype=np.float32)

    # Normalisation L2 finale (comme pyannote)
    if l2_normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)

    if return_times:
        starts_sec = np.asarray(starts_samples, dtype=np.float32) / 16000.0
        return embs, starts_sec
    return embs
