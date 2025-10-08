import numpy as np
from src.asr_jetson.diarization.clustering import cluster_embeddings


def test_cluster_embeddings_two_groups():
    """
    Test clustering with two well-separated groups of embeddings.
    """
    # Génère 2 groupes distincts (2D pour simplifier)
    group1 = np.random.normal(loc=0.0, scale=0.1, size=(5, 2))
    group2 = np.random.normal(loc=5.0, scale=0.1, size=(5, 2))
    embeddings = np.vstack([group1, group2])

    labels = cluster_embeddings(embeddings, n_speakers=2, method="spectral")

    assert labels.shape[0] == embeddings.shape[0]
    # On doit avoir au moins 2 clusters différents
    assert len(set(labels)) == 2


def test_cluster_embeddings_empty_input():
    """
    Test clustering with empty input.
    """
    embeddings = np.zeros((0, 256))  # aucun segment
    labels = cluster_embeddings(embeddings, n_speakers=2, method="spectral")

    assert labels.shape[0] == 0
