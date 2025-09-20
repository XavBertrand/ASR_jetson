import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


def spectral_clustering_torch(
        embeddings: np.ndarray,
        n_speakers: int = 2,
        device: str = "cpu",
        sigma: float = 1.0
) -> np.ndarray:
    """
    Spectral clustering optimisé pour GPU avec PyTorch.
    Plus rapide et stable que sklearn pour la diarization.

    Args:
        embeddings: Speaker embeddings (n_segments, emb_dim)
        n_speakers: Nombre de speakers
        device: "cpu" ou "cuda"
        sigma: Paramètre pour le kernel RBF (plus petit = plus sélectif)
    """
    if embeddings.shape[0] <= 1:
        return np.zeros(embeddings.shape[0], dtype=int)

    n_speakers = min(n_speakers, embeddings.shape[0])

    # Conversion vers PyTorch
    X = torch.from_numpy(embeddings).float().to(device)

    # Normalisation L2
    X = F.normalize(X, p=2, dim=1)

    # Matrice de similarité cosinus (plus stable que RBF pour TitaNet)
    similarity = torch.mm(X, X.t())
    similarity = torch.clamp(similarity, -1.0, 1.0)

    # Conversion en affinité (0 à 1)
    affinity = (similarity + 1.0) / 2.0

    # Laplacien normalisé
    D = torch.diag(torch.sum(affinity, dim=1))
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D) + 1e-8))
    L_norm = torch.eye(affinity.shape[0], device=device) - torch.mm(torch.mm(D_inv_sqrt, affinity), D_inv_sqrt)

    # Eigendecomposition (CPU plus stable pour petites matrices)
    L_cpu = L_norm.cpu().numpy()
    eigenvals, eigenvecs = np.linalg.eigh(L_cpu)

    # Prendre les k plus petites valeurs propres (sauf la première)
    idx = np.argsort(eigenvals)
    embedding_vectors = eigenvecs[:, idx[1:n_speakers + 1]]

    # Normalisation des vecteurs propres
    norms = np.linalg.norm(embedding_vectors, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    embedding_vectors = embedding_vectors / norms

    # K-means sur les vecteurs propres
    kmeans = KMeans(n_clusters=n_speakers, random_state=0, n_init=10)
    labels = kmeans.fit_predict(embedding_vectors)

    return labels


def agglomerative_clustering_cosine(
        embeddings: np.ndarray,
        n_speakers: int = 2,
        threshold: float = 0.8
) -> np.ndarray:
    """
    Clustering hiérarchique ascendant optimisé pour la diarization.
    Plus léger que spectral clustering, très adapté aux embeddings speakers.

    Args:
        embeddings: Speaker embeddings
        n_speakers: Nombre de speakers cible
        threshold: Seuil de similarité cosinus pour merger
    """
    if embeddings.shape[0] <= 1:
        return np.zeros(embeddings.shape[0], dtype=int)

    # Normalisation L2
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    embeddings_norm = embeddings / norms

    # Initialisation : chaque segment = un cluster
    clusters = [[i] for i in range(embeddings.shape[0])]
    centroids = embeddings_norm.copy()

    # Merge jusqu'à avoir n_speakers clusters
    while len(clusters) > n_speakers:
        # Calculer similarités entre tous les centroids
        similarities = cosine_similarity(centroids)
        np.fill_diagonal(similarities, -1)  # Éviter auto-merge

        # Trouver la paire la plus similaire
        max_idx = np.unravel_index(np.argmax(similarities), similarities.shape)
        i, j = max_idx[0], max_idx[1]

        if similarities[i, j] < threshold:
            break  # Arrêter si similarité trop faible

        # Merger les clusters
        clusters[i].extend(clusters[j])

        # Nouveau centroid (moyenne pondérée)
        indices = clusters[i]
        centroids[i] = np.mean(embeddings_norm[indices], axis=0)
        centroids[i] /= (np.linalg.norm(centroids[i]) + 1e-8)

        # Supprimer le cluster j
        del clusters[j]
        centroids = np.delete(centroids, j, axis=0)

    # Créer les labels finaux
    labels = np.zeros(embeddings.shape[0], dtype=int)
    for cluster_id, indices in enumerate(clusters):
        for idx in indices:
            labels[idx] = cluster_id

    return labels


def ahc_linkage_clustering(
        embeddings: np.ndarray,
        n_speakers: int = 2,
        linkage: str = "average"
) -> np.ndarray:
    """
    Version ultra-légère avec linkage clustering.
    Très rapide, adapté pour Jetson Nano.
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
    from scipy.spatial.distance import cosine

    if embeddings.shape[0] <= 1:
        return np.zeros(embeddings.shape[0], dtype=int)

    # Normalisation L2
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    embeddings_norm = embeddings / norms

    # Distance cosinus (1 - similarité cosinus)
    distances = []
    n = embeddings.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine(embeddings_norm[i], embeddings_norm[j])
            distances.append(dist)

    # Linkage clustering
    Z = scipy_linkage(distances, method=linkage)
    labels = fcluster(Z, n_speakers, criterion='maxclust') - 1

    return labels


def cluster_embeddings(
        embeddings: np.ndarray,
        n_speakers: int = 2,
        method: str = "torch_spectral",
        device: str = "cpu"
) -> np.ndarray:
    """
    Fonction principale avec plusieurs méthodes de clustering.

    Args:
        embeddings: Speaker embeddings
        n_speakers: Nombre de speakers
        method: "torch_spectral", "agglomerative", "linkage"
        device: "cpu" ou "cuda"
    """
    if embeddings.shape[0] == 0:
        return np.zeros(0, dtype=int)

    # Nettoyage initial
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

    try:
        if method == "torch_spectral":
            return spectral_clustering_torch(embeddings, n_speakers, device)
        elif method == "agglomerative":
            return agglomerative_clustering_cosine(embeddings, n_speakers)
        elif method == "linkage":
            return ahc_linkage_clustering(embeddings, n_speakers)
        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        print(f"[WARNING] {method} failed: {e}, falling back to KMeans")
        # Fallback vers KMeans
        from sklearn.cluster import KMeans
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        embeddings_norm = embeddings / norms
        kmeans = KMeans(n_clusters=min(n_speakers, embeddings.shape[0]), random_state=0)
        return kmeans.fit_predict(embeddings_norm)


# Alias pour compatibilité
def cluster_speakers(embeddings: np.ndarray, n_speakers: int = 2) -> np.ndarray:
    """Compatible avec votre pipeline existant."""
    return cluster_embeddings(embeddings, n_speakers, method="agglomerative")