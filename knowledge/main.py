from chunking import chunk_text
from embeddings import EmbeddingModel

with open('./data/document.txt','r') as file:
    document = file.read()

chunks = chunk_text(document, chunk_size = 200, overlap=50)

embedding_model = EmbeddingModel()

embeddings = embedding_model.embed_documents(chunks)
print("Number of Chunks: ", len(chunks))
print("Embedding shape: ", embeddings.shape)