import numpy as np
from sklearn.cluster import SpectralClustering


def cluster_speakers(embeddings: np.ndarray, n_speakers: int = 2) -> np.ndarray:
    """
    Cluster embeddings into speakers using spectral clustering.
    """
    if embeddings.shape[0] == 0:
        return np.array([])

    clustering = SpectralClustering(
        n_clusters=n_speakers,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=42,
    )
    return clustering.fit_predict(embeddings)
