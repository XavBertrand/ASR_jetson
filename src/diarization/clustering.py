# src/diarization/clustering.py
from __future__ import annotations

import numpy as np
from typing import Literal, Optional

from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA

DiarizerMethod = Literal["spectral", "kmeans"]


def _cosine_similarity_matrix(
    X: np.ndarray,
    eps: float = 1e-8,
    dtype: type = np.float32,
) -> np.ndarray:
    """
    Calcule une matrice de similarité cosinus (n x n) robuste.

    - Normalisation L2 avec epsilon pour éviter divisions par zéro.
    - Nettoyage NaN/Inf -> 0.
    - Clamp [-1, 1] pour stabilité num.
    - Diagonale = 1.

    Args:
        X: (num_chunks, emb_dim) embeddings.
        eps: petit terme de stabilité lors de la normalisation.
        dtype: dtype cible pour économiser mémoire (float32 par défaut).

    Returns:
        S: (num_chunks, num_chunks) en 'dtype'.
    """
    if X.ndim != 2:
        raise ValueError("embeddings must be a 2D array [num_chunks, emb_dim]")

    X = X.astype(dtype, copy=False)

    # Normalisation L2 sécurisée
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    Xn = X / norms

    # Similarité cosinus
    S = Xn @ Xn.T

    # Nettoyage / Clamp / Diagonale
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(S, -1.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S


def cluster_embeddings(
    embeddings: np.ndarray,
    n_speakers: int,
    method: DiarizerMethod = "spectral",
    spectral_assign_labels: Literal["kmeans", "cluster_qr"] = "kmeans",
    random_state: int = 0,
    pca_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Regroupe des embeddings en `n_speakers` clusters.

    - method="spectral": SpectralClustering(affinity='precomputed', assign_labels=...)
                         sur une matrice de similarité cosinus robuste.
    - method="kmeans":   KMeans directement sur les embeddings.

    Paramètres:
        embeddings      : np.ndarray (num_chunks, emb_dim)
        n_speakers      : nombre de locuteurs (>= 1)
        method          : "spectral" | "kmeans"
        spectral_assign_labels : 'kmeans' (défaut) ou 'cluster_qr'
        random_state    : graine déterministe
        pca_dim         : si défini, PCA -> réduction à pca_dim avant clustering
                          (utile pour du bruit de haute dimension)

    Retour:
        labels: np.ndarray (num_chunks,) d'indices de clusters int.
    """
    # Cas vides / triviaux
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
        # Pas assez de segments -> chacun son cluster
        return np.arange(n_samples, dtype=int)

    X = embeddings
    # Optionnel : réduction de dimension (stabilise parfois Spectral)
    if pca_dim is not None and pca_dim > 0 and X.shape[1] > pca_dim:
        try:
            X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)
        except Exception:
            # PCA facultative : en cas d'échec, on continue sans
            X = embeddings

    if method == "kmeans":
        km = KMeans(n_clusters=n_speakers, n_init="auto", random_state=random_state)
        return km.fit_predict(X)

    # --- method == "spectral" ---
    S = _cosine_similarity_matrix(X)

    # SpectralClustering est parfois sensible numériquement ; paramètres un peu plus stables
    try:
        sc = SpectralClustering(
            n_clusters=n_speakers,
            affinity="precomputed",
            assign_labels=spectral_assign_labels,
            random_state=random_state,
            # n_init est utilisé côté assign_labels='kmeans' (>=1.3); ignoré sinon
            n_init=10 if spectral_assign_labels == "kmeans" else None,
            eigen_solver="arpack",  # généralement stable pour petites tailles
        )
        labels = sc.fit_predict(S)
        return labels.astype(int, copy=False)
    except Exception:
        # Fallback robuste : KMeans direct si Spectral échoue (NaN, graph déconnecté, etc.)
        km = KMeans(n_clusters=n_speakers, n_init="auto", random_state=random_state)
        return km.fit_predict(X).astype(int, copy=False)
