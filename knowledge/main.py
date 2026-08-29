from chunking import chunk_text
from embeddings import EmbeddingModel
from vector_store import *
# 1. Load Document
with open('./data/document.txt','r') as file:
    document = file.read()

# 2. Chunk Document
chunks = chunk_text(document, chunk_size = 200, overlap=50)

# 3. Create Embeddings
embedding_model = EmbeddingModel()


document_embeddings = embedding_model.embed_documents(chunks)

# 4. Store Vectors
vector_store = VectorStore()
vector_store.add(
    chunks,
    document_embeddings
)

# 5. User Query
query = "What does RAG do?"

# 6. Embed Query
query_embedding = (
    embedding_model.embed_query(query)
)

# 7. Retrieve Documents
results = vector_store.search(
    query_embedding,
    top_k = 2
)

# 8. Print Results

for result in results:
    print(f'\n Similarity Score: {result['score']}')
    print("Document")
    print(result['document'])