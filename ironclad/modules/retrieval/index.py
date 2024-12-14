# ironclad/modules/retrieval/index.py

import os
import faiss
import numpy as np
import torch

class EmbeddingIndex:
    def __init__(self, dimension=512):
        """
        Initializes the FAISS index for embedding storage.

        Parameters:
        - dimension (int): Dimensionality of the embeddings (e.g., 512 for InceptionResnetV1).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance metric
        self.image_paths = []  # Keeps track of file paths associated with embeddings

    def add(self, embedding, image_path):
        """
        Adds an embedding and its associated image path to the index.

        Parameters:
        - embedding (torch.Tensor or np.ndarray): Embedding to add.
        - image_path (str): Path of the image associated with the embedding.
        """
        if isinstance(embedding, torch.Tensor):  # Convert to numpy if needed
            embedding = embedding.numpy()
        self.index.add(embedding.reshape(1, -1))
        self.image_paths.append(image_path)

    def save(self, catalog_dir):
        """
        Saves the index and metadata to the specified directory.

        Parameters:
        - catalog_dir (str): Directory to save the index and metadata.
        """
        os.makedirs(catalog_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(catalog_dir, 'embeddings.index'))
        np.save(os.path.join(catalog_dir, 'image_paths.npy'), self.image_paths)

    def load(self, catalog_dir):
        """
        Loads the index and metadata from the specified directory.

        Parameters:
        - catalog_dir (str): Directory containing the saved index and metadata.
        """
        self.index = faiss.read_index(os.path.join(catalog_dir, 'embeddings.index'))
        self.image_paths = np.load(os.path.join(catalog_dir, 'image_paths.npy'), allow_pickle=True).tolist()
