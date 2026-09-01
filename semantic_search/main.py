from sentence_transformers import SentenceTransformer, util
from semantic import SemanticSearch
from similarity import consine_similarity
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

with open('./data/document.txt','r') as file:
    document = file.read()
documents = []
for doc in document.split("\n"):
    documents.append(doc)

search_engine = SemanticSearch(documents)
results = search_engine.search("What is the capital of India?", k=3)
for doc, similarity in results:
    print(f"Document: {doc}, Similarity: {similarity:.4f}")