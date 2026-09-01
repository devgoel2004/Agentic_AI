import chromadb
client = chromadb.PersistentClient(
    path = "./chroma_db"
)

collection = client.create_collection(name="documents")