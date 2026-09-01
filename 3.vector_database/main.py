from chroma import collection
documents = [
    "Python is commonly used for web development and automation.",
    "JavaScript is used to build interactive web applications.",
    "Machine learning allows computers to learn from data.",
    "PostgreSQL is a relational database management system.",
    "Pizza is a popular Italian food."
]
ids = [
    "doc1",
    "doc2",
    "doc3",
    "doc4",
    "doc5"
]
collection.add(
    documents=documents,
    ids=ids
)
query = "How can I build a web application using Python?"
results = collection.query(
    query_texts = [query],
    n_results = 2
)

print(results)