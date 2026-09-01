class TextChunker:
    def __init__(
        self,
        chunk_size : int,
        chunk_overlap: int
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

# We can replace it with a more advanced chunking method like semantic chunking or token-based chunking with overlap. For now, we will use a simple character-based chunking method.
    def chunk(self, text: str)-> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks