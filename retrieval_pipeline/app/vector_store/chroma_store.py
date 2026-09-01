import chromadb

class ChromaVectorStore:
    def __init__(self, path: str, collection_name : str):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.create_collection(name=collection_name)

    def upsert(self, ids, documents, embeddings, metadatas):
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    def search(self, query_embedding, top_k):
        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )
        return results