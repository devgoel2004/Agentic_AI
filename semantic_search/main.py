from sentence_transformers import SentenceTransformer, util
from similarity import consine_similarity
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

with open('./data/document.txt','r') as file:
    document = file.read()
documents = []
for doc in document.split("\n"):
    documents.append(doc)


document_embeddings = model.encode(documents)
print(document_embeddings.shape)
query = "How can I create a website using Python?"
results = []
for doc, embedding in zip(documents, document_embeddings):
    similarity = consine_similarity(
        model.encode(query),
        embedding
    )
    results.append((doc, similarity))

results.sort(
    key=lambda item: item[1],
    reverse=True
)
for document, score in results:
    print(
        f"{score:.4f} → {document}"
    )