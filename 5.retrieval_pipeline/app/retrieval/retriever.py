class Retriever:
    def __init__(self, embedding_model, vector_store, default_top_k: int = 5):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(self, query:str, top_k: int | None = None):
        top_k = (
            top_k
            or self.default_top_k  
        )

        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        return results