import uuid
import hashlib
class DocumentIngestionService:
    def __init__(self, chunker, embedding_model, vector_store):
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def generate_chunk_id(self, source: str, chunk_index: int)->str:
        value = f"{source}_{chunk_index}"
        return hashlib.sha256(
            value.encode('utf-8')
        ).hexdigest()
    
    def ingest(self, text: str, source: str):
        # 1. Chunk the documents
        chunks = self.chunker.chunk(text)

        # 2. Embed the chunks
        embeddings = self.embedding_model.embed_documents(chunks)

        # 3. Create IDs
        ids = [
            self.generate_chunk_id(
                source = source,
                chunk_index = index
            )
            for index in range(len(chunks))
        ]

        # 4. Create metadata
        metadatas = [
            {
                "source":source,
                "chunk_index":index,
            }
            for index in range(len(chunks))
        ]

        # 5. Store everything
        self.vector_store.upsert(ids, chunks, embeddings, metadatas)
        return len(chunks)