from chunking import chunk_text
from embeddings import EmbeddingModel
from vector_store import *
from generator import build_prompt
from chunking import recursive_split, add_overlap, recursive_token_chunk, token_chunk_with_overlap,semantic_chunk
from llm import LLM
from transformers import AutoTokenizer
# 1. Load Document
with open('./data/document.txt','r') as file:
    document = file.read()

with open("./data/python_document.txt",'r') as file:
    document1 = file.read()

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

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
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
# Token based chunks with overlap
# chunks = token_chunk_with_overlap(text=document, tokenizer = tokenizer, chunk_size=500, chunk_overlap=40)

# Semantic Chunking
chunks = semantic_chunk(text=document1,percentile=20)
for chunk in chunks:
    print(chunk)
    print("-"*50)