def chunk_text(text, chunk_size = 200, overlap = 50):
    if(overlap >= chunk_size):
        raise ValueError(
            "overlap must be smaller than chunks"
        )
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
