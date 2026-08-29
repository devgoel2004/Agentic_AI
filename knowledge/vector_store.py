import numpy as np

class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = None

    def add(self, documents, embeddings):
        self.documents.extend(documents)
        embeddings = np.array(embeddings)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([       # Vertical Stack
                self.embeddings,
                embeddings
            ])

    def search(self, query_embedding, top_k = 3):
        query_embedding = np.array(query_embedding)
        # Calculate the dot product
        similarities = np.dot(
            self.embeddings,
            query_embedding
        )
        # Normalize to calculate Cosine Similarity
        similarities = similarities / (
                np.linalg.norm(
                    self.embeddings,
                    axis=1
                )
                *
                np.linalg.norm(query_embedding)
        )
        # This returns the indices in ascending order
        top_indices = np.argsort(
            similarities
        )[-top_k:][::-1]
        results = []
        for index in top_indices:
            results.append({
                "document": self.documents[index],
                "score": similarities[index]
            })
        return results