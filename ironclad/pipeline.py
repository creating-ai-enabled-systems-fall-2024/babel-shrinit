import os
import torch
from modules.extraction.embedding import EmbeddingExtractor
from modules.extraction.processing import Preprocessor
from modules.retrieval.index import EmbeddingIndex

class Pipeline:
    def __init__(self, model_name='casia-webface', device='cpu', dimension=512):
        self.device = device
        self.extractor = EmbeddingExtractor(model_name=model_name)
        self.extractor.model = self.extractor.model.to(device)
        self.preprocessor = Preprocessor()
        self.index = EmbeddingIndex(dimension=dimension)

    def __encode(self, image_path):
        image_tensor = self.preprocessor.preprocess(image_path).to(self.device)
        embedding = self.extractor.extract(image_tensor)
        return embedding

    def __precompute(self, gallery_directory):
        for person_name in os.listdir(gallery_directory):
            person_dir = os.path.join(gallery_directory, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    try:
                        embedding = self.__encode(image_path)
                        self.index.add(embedding, image_path)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

    def __save_embeddings(self, catalog_dir='storage/catalog'):
        self.index.save(catalog_dir)

    def search_gallery(self, probe_image_path, k=5):
        probe_embedding = self.__encode(probe_image_path)
        search_results = self.index.index.search(probe_embedding.unsqueeze(0).cpu().numpy(), k)
        distances, indices = search_results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            image_path = self.index.image_paths[idx]
            person_name = os.path.basename(os.path.dirname(image_path))
            results.append({
                'name': person_name,
                'image_path': image_path,
                'distance': distances[0][i]
            })
        return results
