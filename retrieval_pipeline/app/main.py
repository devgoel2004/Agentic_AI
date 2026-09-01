from app.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TOP_K
)

from app.ingestion.chunker import TextChunker

from app.embeddings.embedding_model import (
    EmbeddingModel
)

from app.vector_store.chroma_store import (
    ChromaVectorStore
)

from app.ingestion.ingest import (
    DocumentIngestionService
)

from app.retrieval.retriever import (
    Retriever
)

chunker = TextChunker(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP
)

embedding_model = EmbeddingModel()

vector_store = ChromaVectorStore(
    collection_name = COLLECTION_NAME,
    path = str(CHROMA_PATH),
    collection_name = COLLECTION_NAME
)

ingestion_service = DocumentIngestionService(
    chunker = chunker,
    embedding_model = embedding_model,
    vector_store = vector_store
)

retriever = Retriever(
    embedding_model = embedding_model,
    vector_store = vector_store,
    default_top_k = DEFAULT_TOP_K
)


document = """
Python is widely used for web development.

Django and FastAPI are popular frameworks
for building Python web applications.

Python is also commonly used for artificial
intelligence and machine learning.
"""

ingestion_service.ingest(text=document,source="python_guide")

results = retriever.retrieve(query="What is a popular programming language for web development?", top_k=2)

print(results['documents'][0])