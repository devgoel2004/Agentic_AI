import math
from sentence_transformers import SentenceTransformer
class SemanticSearch:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.document_embeddings = self.model.encode(documents)

    def consine_similarity(self, vector_a, vector_b):
        dot_product = sum(
            a*b
            for a, b in zip(vector_a, vector_b)
        )
        magnitude_a = math.sqrt(sum(a**2 for a in vector_a))
        magnitude_b = math.sqrt(sum(b**2 for b in vector_b))
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        similarity = dot_product / (magnitude_a * magnitude_b)
        return similarity
    def search(self, query, k=3):
        query_embedding = self.model.encode(query)
        results = []
        for doc, embedding in zip(self.documents, self.document_embeddings):
            similarity = self.consine_similarity(
                query_embedding,
                embedding
            )
            results.append((doc, similarity))
        results.sort(
            key=lambda item: item[1],
            reverse=True
        )
        return results[:k]