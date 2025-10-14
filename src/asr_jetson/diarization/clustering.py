# src/diarization/clustering.py
from __future__ import annotations

import numpy as np
from typing import Literal, Optional

from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.signal import medfilt

DiarizerMethod = Literal["spectral", "kmeans", "hierarchical"]


def _cosine_similarity_matrix(
        X: np.ndarray,
        eps: float = 1e-8,
        dtype: type = np.float32,
) -> np.ndarray:
    """Calcule une matrice de similarité cosinus robuste."""
    if X.ndim != 2:
        raise ValueError("embeddings must be a 2D array [num_chunks, emb_dim]")

    X = X.astype(dtype, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    Xn = X / norms

    S = Xn @ Xn.T
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(S, -1.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S


def _segments_from_labels(lbls):
    segs = []
    if len(lbls) == 0: return segs
    cur_l, cur_s = lbls[0], 0
    for i, l in enumerate(lbls[1:], 1):
        if l != cur_l:
            segs.append([cur_s, i, cur_l])
            cur_l, cur_s = l, i
    segs.append([cur_s, len(lbls), cur_l])
    return segs


def smooth_labels(labels, win_sec=1.5, hop_sec=2.0, min_dur_sec=1.0):
    """Lissage OPTIONNEL - désactiver si les résultats sont meilleurs sans."""
    # Filtre médian très léger (kernel=3 au lieu de 5)
    labels = medfilt(labels, kernel_size=3)
    min_len_frames = max(1, int((min_dur_sec - (win_sec - hop_sec)) / hop_sec))

    segs = _segments_from_labels(labels.copy())
    for k in range(len(segs)):
        s, e, l = segs[k]
        if (e - s) < min_len_frames:
            if k == 0:
                segs[k][2] = segs[k + 1][2]
            elif k == len(segs) - 1:
                segs[k][2] = segs[k - 1][2]
            else:
                lprev = segs[k - 1][2]
                lnext = segs[k + 1][2]
                segs[k][2] = lprev if (segs[k - 1][1] - segs[k - 1][0]) >= (segs[k + 1][1] - segs[k + 1][0]) else lnext

    labels_smooth = np.empty_like(labels)
    for s, e, l in segs:
        labels_smooth[s:e] = l

    return labels_smooth


def cluster_embeddings(
        embeddings: np.ndarray,
        n_speakers: int,
        method: DiarizerMethod = "spectral",
        spectral_assign_labels: Literal["kmeans", "cluster_qr"] = "kmeans",
        random_state: int = 0,
        pca_dim: Optional[int] = None,
        apply_smoothing: bool = False,  # NOUVEAU : désactive par défaut
) -> np.ndarray:
    """
    Regroupe des embeddings en `n_speakers` clusters.

    CHANGEMENT CLÉ :
    - Hierarchical clustering corrigé (affinity='cosine' sur embeddings, pas sur similarité)
    - Lissage désactivé par défaut (apply_smoothing=False)
    """
    # Cas triviaux
    if embeddings is None or len(embeddings) == 0:
        return np.array([], dtype=int)
    if n_speakers <= 0:
        raise ValueError("n_speakers must be >= 1")
    n_samples = len(embeddings)
    if n_samples == 1:
        return np.array([0], dtype=int)
    if n_speakers == 1:
        return np.zeros(n_samples, dtype=int)
    if n_samples < n_speakers:
        return np.arange(n_samples, dtype=int)

    X = embeddings

    # PCA optionnel
    if pca_dim is not None and pca_dim > 0 and X.shape[1] > pca_dim:
        try:
            X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)
        except Exception:
            X = embeddings

    # === KMEANS : simple et efficace ===
    if method == "kmeans":
        km = KMeans(n_clusters=n_speakers, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        return labels.astype(int)

    # === HIERARCHICAL : CORRIGÉ ===
    if method == "hierarchical":
        try:
            # AgglomerativeClustering avec affinity='cosine' sur les EMBEDDINGS directement
            agg = AgglomerativeClustering(
                n_clusters=n_speakers,
                metric='cosine',  # ou affinity='cosine' selon ta version sklearn
                linkage='average'
            )
            labels = agg.fit_predict(X)

            # Lissage OPTIONNEL
            if apply_smoothing:
                labels = smooth_labels(labels)

            return labels.astype(int)
        except Exception as e:
            print(f"[WARN] Hierarchical clustering failed: {e}, falling back to KMeans")
            km = KMeans(n_clusters=n_speakers, n_init=10, random_state=random_state)
            return km.fit_predict(X).astype(int)

    # === SPECTRAL : meilleur pour les cas complexes ===
    S = _cosine_similarity_matrix(X)

    try:
        sc = SpectralClustering(
            n_clusters=n_speakers,
            affinity="precomputed",
            assign_labels=spectral_assign_labels,
            random_state=random_state,
            n_init=10,
            eigen_solver="arpack",
        )
        labels = sc.fit_predict(S)

        # Lissage OPTIONNEL
        if apply_smoothing:
            labels = smooth_labels(labels)

        return labels.astype(int)
    except Exception as e:
        print(f"[WARN] Spectral clustering failed: {e}, falling back to KMeans")
        km = KMeans(n_clusters=n_speakers, n_init=10, random_state=random_state)
        return km.fit_predict(X).astype(int)