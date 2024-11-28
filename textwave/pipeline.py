import os
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.indexing import FaissIndex


class Pipeline:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.embedding = Embedding(model_name=embedding_model)
        self.index = FaissIndex(index_type='Flat')  

    def preprocess_corpus(self, corpus_directory, chunking_strategy, fixed_length=None, overlap_size=2):
        processor = DocumentProcessing()
        chunks_metadata = []

        for filename in os.listdir(corpus_directory):
            filepath = os.path.join(corpus_directory, filename)
            if not filename.endswith('.txt.clean'):
                continue

            if chunking_strategy == 'sentence':
                chunks = processor.sentence_chunking(filepath, overlap_size=overlap_size)
            elif chunking_strategy == 'fixed-length' and fixed_length is not None:
                chunks = processor.fixed_length_chunking(filepath, fixed_length=fixed_length, overlap_size=overlap_size)
            else:
                raise ValueError("Invalid chunking strategy")

            embeddings = self.embedding.encode(chunks) 
            metadata = [{"filename": filename, "chunk": chunk} for chunk in chunks]
            self.index.add_embeddings(embeddings, metadata) 

            chunks_metadata.extend(metadata)

        return chunks_metadata
