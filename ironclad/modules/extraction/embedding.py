# ironclad/modules/extraction/embedding.py

import torch
from facenet_pytorch import InceptionResnetV1

class EmbeddingExtractor:
    def __init__(self, model_name='casia-webface'):
        """
        Initializes the embedding model.

        Parameters:
        - model_name (str): Pre-trained model name ('casia-webface' or 'vggface2').
        """
        self.model = InceptionResnetV1(pretrained=model_name).eval()  # Load pre-trained model

    def extract(self, image_tensor):
        """
        Extracts embeddings from a preprocessed image tensor.

        Parameters:
        - image_tensor (torch.Tensor): Preprocessed image tensor.

        Returns:
        - embedding (torch.Tensor): Embedding vector.
        """
        with torch.no_grad():
            return self.model(image_tensor.unsqueeze(0)).squeeze(0)
