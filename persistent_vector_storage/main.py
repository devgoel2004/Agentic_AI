from chroma import collection

documents = [
    "Python is commonly used for web development and automation.",
    "JavaScript is used to build interactive web applications.",
    "PostgreSQL is a relational database management system."
]
metadatas = [
    {
        "category": "programming",
        "source": "python_guide"
    },
    {
        "category": "programming",
        "source": "javascript_guide"
    },
    {
        "category": "database",
        "source": "postgresql_guide"
    }
]
collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3"],
    metadatas=metadatas
)

query = "What is a popular programming language for web development?"
results = collection.query(
    query_texts = query,
    n_results = 2,
    where = {
        "category":"programming"
    }
)

print(results["documents"])