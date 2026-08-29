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

def split_by_separator(text, separator):
    """
    Split text using a seperator.

    If separator is an empty string,
    split into individual characters.
    """
    if separator == "":
        return list(text)
    return text.split(separator)

def recursive_chunk(text, chunk_size = 300, separators = None):
    if separators is None:
        separators = [
            "\n\n",  # paragraphs
            ". ",    # sentences
            " ",     # words
            ""       # characters
        ]

    if len(text)<=chunk_size:
        return [text]

    separator = separators[0]
    remaining_separators = separators[1:]
    pieces = split_by_separator(
        text,
        separator
    )
    chunks = []
    current_chunk = ""
    for piece in pieces:
        if current_chunk:
            candidate = (
                current_chunk + separator + piece
            )
        else:
            candidate = piece

    if len(candidate)<=chunk_size:
        current_chunk = candidate
    else:
        if current_chunk:
            chunks.append(current_chunk)

        if len(piece) > chunk_size:
            smaller_chunks = recursive_chunk(
                piece,
                chunk_size,
                remaining_separators
            )
            chunks.extend(smaller_chunks)
            current_chunk = ""

        else:
            current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def add_overlap(chunks, overlap = 50):
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
        else:
            previous_chunk = chunks[i - 1]

            overlap_text = previous_chunk[-overlap:]

            new_chunk = (
                overlap_text
                + "\n"
                + chunk
            )

            overlapped_chunks.append(new_chunk)

    return overlapped_chunks