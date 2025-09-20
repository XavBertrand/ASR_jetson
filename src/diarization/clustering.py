# src/diarization/clustering.py
from __future__ import annotations
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from typing import Literal

DiarizerMethod = Literal["spectral", "kmeans"]

def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Cosine similarity (n x n) robuste : normalisation + NaN->0 + diag=1.
    """
    if X.ndim != 2:
        raise ValueError("embeddings must be a 2D array [num_chunks, emb_dim]")
    # normalisation L2
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # éviter la division par zéro
    norms[norms == 0] = 1.0
    Xn = X / norms
    S = Xn @ Xn.T
    # Nettoyage
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    # clamp léger pour stabilité numérique
    np.clip(S, -1.0, 1.0, out=S)
    # diagonale = 1
    np.fill_diagonal(S, 1.0)
    return S

def cluster_embeddings(
    embeddings: np.ndarray,
    n_speakers: int,
    method: DiarizerMethod = "spectral",
    spectral_assign_labels: Literal["kmeans", "cluster_qr"] = "kmeans",
    random_state: int = 0,
) -> np.ndarray:
    """
    Regroupe des embeddings en n_speakers clusters.

    - method="spectral": SpectralClustering(affinity='precomputed', assign_labels=...)
    - method="kmeans":   KMeans directement sur les embeddings
    """
    if embeddings is None or len(embeddings) == 0:
        return np.array([], dtype=int)

    if n_speakers <= 0:
        raise ValueError("n_speakers must be >= 1")

    if len(embeddings) < n_speakers:
        # pas assez de segments pour autant de speakers : labels uniques
        return np.arange(len(embeddings), dtype=int)

    if method == "kmeans":
        km = KMeans(n_clusters=n_speakers, n_init="auto", random_state=random_state)
        return km.fit_predict(embeddings)

    # method == "spectral"
    S = _cosine_similarity_matrix(embeddings)

    try:
        sc = SpectralClustering(
            n_clusters=n_speakers,
            affinity="precomputed",
            assign_labels=spectral_assign_labels,  # <— param exposé
            random_state=random_state,
        )
        labels = sc.fit_predict(S)
        return labels
    except Exception:
        # Fallback robuste si la matrice pose problème : KMeans sur les embeddings
        km = KMeans(n_clusters=n_speakers, n_init="auto", random_state=random_state)
        return km.fit_predict(embeddings)
