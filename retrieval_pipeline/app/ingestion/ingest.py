import uuid

class DocumentIngestionService:
    def __init__(self, chunker, embedding_model, vector_store):
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def ingest(self, text: str, source: str):
        # 1. Chunk the documents
        chunks = self.chunker.chunk(text)

        # 2. Embed the chunks
        embeddings = self.embedding_model.embed_documents(chunks)

        # 3. Create IDs
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

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