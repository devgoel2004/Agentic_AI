import math
documents = {
    "Python web development": [0.2, 0.8],
    "Java programming": [0.7, 0.3],
    "Italian pizza recipe": [-0.5, 0.1]
}
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a*b
        for a, b in zip(vector_a, vector_b)
    )
    magnitude_a = math.sqrt(sum(a**2 for a in vector_a))
    magnitude_b = math.sqrt(sum(b**2 for b in vector_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity
user_query = [0.3, 0.9]

def similarity_search(query_vector, documents, k=3):
    results = []
    for  doc, doc_vector in documents.items():
        similarity = cosine_similarity(query_vector, doc_vector)
        results.append((doc, similarity))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

result = similarity_search(user_query, documents, k=3)
for doc, similarity in result:
    print(f"Document: {doc}, Similarity: {similarity:.4f}")