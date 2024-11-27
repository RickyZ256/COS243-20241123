from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle 

class EmbeddingGenerator:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

class VectorIndex:
    def __init__(self):
        self.index = []
        self.metadata = []

    def add_to_index(self, embeddings, metadata):
        self.index.append(embeddings)
        self.metadata.extend(metadata)

    def build_index(self):
        self.index = np.vstack(self.index)  # Combine all embeddings into one array

    def search(self, query_embedding, top_k=5):
        print("Performing semantic search...")
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.index)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k] 
        results = [
            {"metadata": self.metadata[i], "similarity": similarities[i]} for i in top_indices
        ]
        return results

    def save_index(self, filepath="vector_index.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump({"index": self.index, "metadata": self.metadata}, f)

    def load_index(self, filepath="vector_index.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.metadata = data["metadata"]

if __name__ == "__main__":
    # Assuming chunked_documents is from Step 2
    chunked_texts = [doc.text for doc in chunked_documents]
    chunked_metadata = [doc.metadata for doc in chunked_documents]

    # Generate embeddings
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(chunked_texts)

    # Build vector index
    vector_index = VectorIndex()
    vector_index.add_to_index(embeddings, chunked_metadata)
    vector_index.build_index()

    # Save index for later use
    vector_index.save_index("vector_index.pkl")

    # Perform a semantic search
    query = "Search for a concept in my eBooks"
    query_embedding = embedding_generator.generate_embeddings([query])[0]
    results = vector_index.search(query_embedding, top_k=5)

    # Display search results
    for result in results:
        print(f"File: {result['metadata']['file_name']}, Similarity: {result['similarity']:.4f}")