import os
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.indexing import FaissIndex
from modules.retrieval.reranker import Reranker
from modules.generator.question_answering import QA_Generator

class Pipeline:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', index_type='Flat', 
                 reranker_type='hybrid', generator_api_key=None):
        """
        Initializes the Pipeline class with embedding, indexing, reranking, and question-answering strategies.
        
        :param embedding_model: The embedding model name to use for encoding queries.
        :param index_type: The type of Faiss index to use ('Flat', 'IVF', etc.).
        :param reranker_type: Reranking strategy ('cross_encoder', 'tfidf', or 'hybrid').
        :param generator_api_key: API key for the question-answer generator (e.g., Mistral).
        """
        self.embedding = Embedding(model_name=embedding_model)
        self.index = FaissIndex(index_type=index_type)
        self.reranker = Reranker(type=reranker_type)
        self.qa_generator = QA_Generator(api_key=generator_api_key)

    def preprocess_corpus(self, corpus_directory, chunking_strategy, fixed_length=None, overlap_size=2):
        """
        Preprocesses the corpus by chunking, embedding, and adding to the index.
        
        :param corpus_directory: Path to the directory containing corpus files.
        :param chunking_strategy: Strategy for chunking ('sentence' or 'fixed-length').
        :param fixed_length: Fixed length for chunks (if using 'fixed-length').
        :param overlap_size: Overlap size for chunks (default is 2).
        :return: Metadata for the chunks added to the index.
        """
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

    def __encode(self, query):
        """
        Encodes the query into an embedding vector.
        
        :param query: User query string.
        :return: Embedding vector for the query.
        """
        query_embedding = self.embedding.encode([query])
        return query_embedding

    def search_neighbors(self, query_embedding, k=10):
        """
        Searches the index for the k-nearest neighbors to the query embedding.
        
        :param query_embedding: Embedding vector of the query.
        :param k: Number of nearest neighbors to retrieve.
        :return: Metadata and distances of the k-nearest neighbors.
        """
        distances, indices, metadata = self.index.search(query_embedding, k)
        return distances, indices, metadata

    def generate_answer(self, query, k=10, rerank=True):
        """
        Generates an answer to the query based on the context from nearest neighbors.
        """
        # Step 1: Encode the query and search for nearest neighbors
        query_embedding = self.__encode(query)
        distances, indices, metadata = self.search_neighbors(query_embedding, k=k)

        # Flatten the nested metadata list
        flat_metadata = [item for sublist in metadata for item in sublist]

        # Extract chunks from the flattened metadata
        context = [meta['chunk'] for meta in flat_metadata if 'chunk' in meta]

        # Step 2: Optionally rerank the context
        if rerank:
            context, _, _ = self.reranker.rerank(query, context)

        # Step 3: Generate the answer using the question-answering generator
        answer = self.qa_generator.generate_answer(query, context[:k])  # Use top-k context
        return answer

    


if __name__ == "__main__":
    # Example usage
    pipeline = Pipeline()

    # Load and preprocess the corpus
    corpus_directory = "storage/corpus"
    pipeline.preprocess_corpus(corpus_directory, chunking_strategy='sentence', overlap_size=2)

    # Test nearest neighbors search
    query = "Who was Abraham Lincoln?"
    query_embedding = pipeline._Pipeline__encode(query)  # Access the private method
    distances, indices, metadata = pipeline.search_neighbors(query_embedding, k=15)

    print("\nQuery:", query)
    print("\nNearest Neighbors:")
    for i, (dist, meta) in enumerate(zip(distances[0], metadata)):
        print(f"Neighbor {i + 1}: Distance {dist}, Metadata: {meta}")
