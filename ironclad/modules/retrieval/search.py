# ironclad/modules/retrieval/search.py

import numpy as np
import torch

class Searcher:
    def __init__(self, embedding_index):
        """
        Initializes the searcher with a given embedding index.

        Parameters:
        - embedding_index (EmbeddingIndex): An instance of the EmbeddingIndex class.
        """
        self.embedding_index = embedding_index

    def search(self, probe_embedding, k=5):
        """
        Searches for the k nearest neighbors of a given probe embedding.

        Parameters:
        - probe_embedding (torch.Tensor or np.ndarray): Embedding of the probe image.
        - k (int): Number of nearest neighbors to return.

        Returns:
        - results (list of dict): List of dictionaries containing 'image_path' and 'distance'.
        """
        if isinstance(probe_embedding, torch.Tensor):
            probe_embedding = probe_embedding.numpy()
        distances, indices = self.embedding_index.index.search(probe_embedding.reshape(1, -1), k)
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'image_path': self.embedding_index.image_paths[idx],
                'distance': distances[0][i]
            })
        return results
