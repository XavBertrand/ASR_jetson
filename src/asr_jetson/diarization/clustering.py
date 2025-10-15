# src/diarization/clustering.py
from __future__ import annotations

import numpy as np
from typing import Literal, Optional

from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.signal import medfilt

from dataclasses import dataclass
from typing import Tuple, Sequence
from sklearn.metrics import silhouette_score
from math import log
from typing import Optional, Tuple


DiarizerMethod = Literal["spectral", "kmeans", "hierarchical", "pyannote"]


def _safe_cosine_distance_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Calcule une matrice de distance cosinus robuste aux vecteurs nuls.
    - Si l'un des deux vecteurs est nul (norme < eps), on met sim=0 (donc dist=1),
      sauf sur la diagonale (dist=0).
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    valid = (norms > eps).astype(np.float32)
    # normalisation sûre (évite division par zéro)
    Xn = X / np.maximum(norms, eps)

    # similarités valides
    S = Xn @ Xn.T  # [-1..1]
    # paires invalides (au moins un zéro) -> sim=0
    mask_invalid = (valid @ valid.T) < 1.0
    S[mask_invalid] = 0.0

    # distance = 1 - sim, diag=0
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D

# --- AJOUT: petite utilité ---
def _cosine_distance_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    S = _cosine_similarity_matrix(X, eps=eps)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D

@dataclass
class AHCParams:
    linkage: str = "average"
    # si n_speakers non fourni, on coupe le dendrogramme par seuil
    distance_threshold: Optional[float] = None  # e.g. 0.35–0.55 (à tuner)

# --- AJOUT: coupe AHC par seuil OU sélection auto du seuil ---
def _ahc_with_threshold(
    X: np.ndarray,
    n_speakers: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    linkage: str = "average",
) -> np.ndarray:
    """
    AHC cosine + linkage average. Si distance_threshold est fourni ET n_speakers=None,
    on laisse sklearn déterminer les clusters par coupe du dendrogramme.
    """
    # on pré-calcule une matrice de distances cosinus "safe"
    D = _safe_cosine_distance_matrix(X)
    kwargs = dict(linkage=linkage, metric="precomputed")

    if distance_threshold is not None and n_speakers is None:
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, **kwargs)
        return agg.fit_predict(D).astype(int)
    else:
        agg = AgglomerativeClustering(n_clusters=n_speakers, **kwargs)
        return agg.fit_predict(D).astype(int)

# --- AJOUT: recherche de seuil "auto" (silhouette sur coupes candidates) ---
def _auto_threshold_via_silhouette(
    X: np.ndarray,
    candidates: Sequence[float] = (0.35, 0.4, 0.45, 0.5, 0.55),
    min_clusters: int = 2,
    max_clusters: int = 8,
) -> float:
    """
    Balaye des seuils de distance cosine, retient celui qui maximise la silhouette
    (et qui produit un nombre sensé de clusters).
    """
    best_thr, best_score = None, -1.0
    for thr in candidates:
        labels = _ahc_with_threshold(X, n_speakers=None, distance_threshold=thr)
        k = len(np.unique(labels))
        if k < min_clusters or k > max_clusters:
            continue
        try:
            # score = silhouette_score(X, labels, metric="cosine")
            # filtre vecteurs non nuls
            eps = 1e-8
            norms = np.linalg.norm(X, axis=1)
            valid_idx = np.where(norms > eps)[0]
            if len(valid_idx) < 2 or len(np.unique(labels)) < 2:
                continue
            Xv = X[valid_idx] / norms[valid_idx, None]
            lv = labels[valid_idx]
            score = silhouette_score(Xv, lv, metric="cosine")


        except Exception:
            continue
        if score > best_score:
            best_score, best_thr = score, thr
    # fallback raisonnable
    return best_thr if best_thr is not None else 0.45

# --- AJOUT: ré-segmentation Viterbi (mono-speaker par frame/segment) ---
def _viterbi_rescore(
    X: np.ndarray,
    labels: np.ndarray,
    penalty_switch: float = 3.0,
    temperature: float = 0.2,
) -> np.ndarray:
    """
    Ré-segmentation légère:
    - on calcule des centroides par cluster initial
    - score d’émission = sim_cosine(x, centroid[label]) / temperature
    - coût de transition: +penalty_switch quand on change de label
    Viterbi renvoie une séquence de labels plus lisse et cohérente temporellement.
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1

    # centroids
    cents = np.zeros((K, X.shape[1]), dtype=np.float32)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            cents[k] = 0.0
        else:
            v = X[idx].mean(axis=0)
            n = np.linalg.norm(v) + 1e-8
            cents[k] = v / n

    # logits d’émission: sim_cosine(x, centroids)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    emis = Xn @ cents.T  # [T, K]
    emis = emis / max(1e-6, temperature)

    T = X.shape[0]
    dp = np.full((T, K), -1e9, dtype=np.float32)
    bk = np.full((T, K), -1, dtype=np.int32)

    # init
    dp[0] = emis[0]

    # dyn prog
    for t in range(1, T):
        for j in range(K):
            # rester sur j
            stay = dp[t-1, j]
            # venir d’un autre i
            change = dp[t-1] - penalty_switch
            prev = np.maximum(stay, change.max())
            dp[t, j] = emis[t, j] + prev
            bk[t, j] = int(stay >= change.max()) and j or int(np.argmax(dp[t-1]))

    # backtrace
    y = np.zeros(T, dtype=int)
    y[-1] = int(np.argmax(dp[-1]))
    for t in range(T-2, -1, -1):
        y[t] = bk[t+1, y[t+1]]
    return y

# --- AJOUT: fusion des labels consécutifs identiques ---
def merge_consecutive_labels(labels: np.ndarray) -> np.ndarray:
    return labels  # (tes exports par tours font déjà la fusion au moment de l’écriture)

# --- AJOUT PRINCIPAL: mode "ahc_viterbi" ---
def cluster_embeddings_pyannote_like(
    embeddings: np.ndarray,
    n_speakers: Optional[int] = None,
    ahc_params: Optional[AHCParams] = None,
    auto_threshold: bool = True,
    viterbi: bool = True,
    penalty_switch: float = 3.0,
    temperature: float = 0.2,
    pca_dim: Optional[int] = None,
    random_state: int = 0,
) -> np.ndarray:
    """
    Pipeline clustering + ré-segmentation "pyannote-like":
    1) AHC global (cosine, linkage average) avec seuil (ou n_speakers si imposé)
    2) Viterbi resegmentation avec pénalité de switch
    """
    if embeddings is None or len(embeddings) == 0:
        return np.array([], dtype=int)

    X = embeddings
    if pca_dim is not None and pca_dim > 0 and X.shape[1] > pca_dim:
        try:
            X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)
        except Exception:
            X = embeddings

    if ahc_params is None:
        ahc_params = AHCParams()

    # Seuil auto si demandé et n_speakers pas imposé
    thr = ahc_params.distance_threshold
    if auto_threshold and n_speakers is None:
        thr = _auto_threshold_via_silhouette(X)

    labels0 = _ahc_with_threshold(
        X, n_speakers=n_speakers, distance_threshold=thr, linkage=ahc_params.linkage
    )

    if viterbi and len(np.unique(labels0)) >= 2:
        labels = _viterbi_rescore(X, labels0, penalty_switch=penalty_switch, temperature=temperature)
    else:
        labels = labels0

    return labels.astype(int)



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
        apply_smoothing: bool = False,
        # Nouveaux paramètres pyannote
        threshold: Optional[float] = 1.1,
        linkage_method: str = "average",
        min_cluster_size: int = 1,
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

    if method == "ahc_viterbi":
        labels = cluster_embeddings_pyannote_like(
            embeddings,
            n_speakers=n_speakers if n_speakers > 0 else None,  # None => seuil auto
            ahc_params=AHCParams(linkage="average", distance_threshold=None),  # on laissera auto
            auto_threshold=(n_speakers <= 0 or n_speakers is None),
            viterbi=True,
            penalty_switch=3.0,
            temperature=0.2,
            pca_dim=pca_dim,
            random_state=random_state,
        )
        return labels.astype(int)

    # === HIERARCHICAL PYANNOTE-LIKE ===
    if method == "hierarchical":
        from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
        from scipy.spatial.distance import cdist

        # Nettoyage
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(X_clean, axis=1, keepdims=True)
        valid_mask = (norms.squeeze(-1) > 1e-8)
        if np.sum(valid_mask) < 2:
            return np.zeros(len(X), dtype=int)

        X_valid = X_clean[valid_mask]
        # Normalisation L2 (comme pyannote)
        X_unit = X_valid / (np.linalg.norm(X_valid, axis=1, keepdims=True) + 1e-12)

        # Dendrogramme euclidien sur vecteurs unitaires
        dendrogram = scipy_linkage(X_unit, method=linkage_method, metric="euclidean")

        labels_valid = None

        # 1) Si n_speakers est fixé -> impose K exact (maxclust)
        if n_speakers is not None and n_speakers > 0:
            labels_valid = fcluster(dendrogram, n_speakers, criterion="maxclust") - 1

        # 2) Sinon, coupe au seuil (distance euclidienne)
        else:
            # ATTENTION: threshold attendu en euclidien
            # Si tu pensais "cosine threshold", utilise: thr_euclid = math.sqrt(2 * thr_cosine)
            thr_euclid = threshold if threshold is not None else 1.1  # défaut raisonnable
            labels_valid = fcluster(dendrogram, thr_euclid, criterion="distance") - 1

        # Gestion min_cluster_size: réassigner les petits -> grand le + proche
        unique, counts = np.unique(labels_valid, return_counts=True)
        large = unique[counts >= max(1, min_cluster_size)]
        small = unique[counts < max(1, min_cluster_size)]

        if len(large) == 0:
            labels_valid[:] = 0
        elif len(small) > 0:
            # centroïdes sur X_unit (cohérent avec cosine/euclid)
            large_centroids = np.vstack([X_unit[labels_valid == k].mean(axis=0) for k in large])
            small_centroids = np.vstack([X_unit[labels_valid == k].mean(axis=0) for k in small])
            # plus proche grand cluster en COSINE (ou euclid, équivalent ici)
            dist = cdist(small_centroids, large_centroids, metric="cosine")
            nearest = np.argmin(dist, axis=1)
            for i, k_large in enumerate(nearest):
                labels_valid[labels_valid == small[i]] = large[k_large]

        # Renumérotation compacte 0..K-1
        _, labels_valid = np.unique(labels_valid, return_inverse=True)

        # Reconstruction globale (met -1 aux embeddings invalides)
        labels = -1 * np.ones(len(X), dtype=int)
        labels[valid_mask] = labels_valid

        if apply_smoothing:
            labels = smooth_labels(labels)

        return labels.astype(int)

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