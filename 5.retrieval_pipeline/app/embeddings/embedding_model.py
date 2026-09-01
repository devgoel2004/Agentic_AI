from sentence_transformers import SentenceTransformer
class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
# Use to embed the document
    def embed_documents(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True)
# Use to embed the query
    def embed_query(self, query:str):
        return self.model.encode(query, normalize_embeddings=True)