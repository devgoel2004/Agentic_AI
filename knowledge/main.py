from chunking import chunk_text
from embeddings import EmbeddingModel
from vector_store import *
from generator import build_prompt
from chunking import recursive_split, add_overlap
from llm import LLM
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
query = "What is the capital of India?"

# 6. Embed Query
query_embedding = (
    embedding_model.embed_query(query)
)

# 7. Retrieve Documents
results = vector_store.search(
    query_embedding,
    top_k = 2
)

# 8. Build Augmented Prompt
prompt = build_prompt(
    query,
    results
)


# Call LLM
response = LLM.generate(LLM,prompt)
print(response)

separators = [
    "\n\n",
    "\n",
    ". ",
    " ",
    ""
]

chunks = recursive_split(text=document, separators=separators, chunk_size=150, chunk_overlap=40)
for i, chunk in enumerate(chunks):
    print("\n" + "=" * 50)
    print(f"CHUNK {i + 1}")
    print(f"Length: {len(chunk)}")
    print("=" * 50)
    print(chunk)