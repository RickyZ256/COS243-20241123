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
        self.index = np.vstack(self.index) 

    def search(self, query_embedding, top_k=5):
        print("Performing semantic search...")

        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.index)

        top_k_indices = similarities.argsort()[0][-top_k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'index': idx,
                'score': similarities[0][idx]
            })
        return results

    def save_index(self, filename="vector_index.pkl"):

        with open(filename, 'wb') as f:
            pickle.dump((self.index, self.metadata), f)

    def load_index(self, filename="vector_index.pkl"):

        with open(filename, 'rb') as f:
            self.index, self.metadata = pickle.load(f)