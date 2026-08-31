from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 1. character Based Chunking
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

#2. Recursive Based Chunking
def merge_splits(
    splits,
    separator,
    chunk_size,
    chunk_overlap
):
    chunks = []
    current_chunk = []
    current_length = 0

    separator_length = len(separator)

    for split in splits:

        split = split.strip()

        if not split:
            continue

        split_length = len(split)

        if current_chunk:
            new_length = (
                current_length
                + separator_length
                + split_length
            )
        else:
            new_length = split_length

        # Add if it fits
        if new_length <= chunk_size:

            current_chunk.append(split)
            current_length = new_length

        else:

            # Save current chunk
            if current_chunk:
                chunks.append(
                    separator.join(current_chunk)
                )

            # Create overlap
            overlap_chunk = []
            overlap_length = 0

            for item in reversed(current_chunk):

                item_length = len(item)

                if overlap_chunk:
                    item_length += separator_length

                if (
                    overlap_length + item_length
                    <= chunk_overlap
                ):
                    overlap_chunk.insert(0, item)

                    overlap_length += item_length

                else:
                    break

            # Start new chunk with overlap
            current_chunk = overlap_chunk

            current_length = len(
                separator.join(current_chunk)
            )

            # Add the current split
            if current_chunk:
                current_length += separator_length

            current_chunk.append(split)
            current_length += split_length

    # Add final chunk
    if current_chunk:
        chunks.append(
            separator.join(current_chunk)
        )

    return chunks


def recursive_split(text,separators,chunk_size=500,chunk_overlap=100):
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    # Text already fits
    if len(text) <= chunk_size:
        return [text.strip()]

    # No separators left
    if not separators:
        return [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

    separator = separators[0]

    # Character-level fallback
    if separator == "":
        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunks.append(
                text[start:end]
            )

            start = end - chunk_overlap

        return chunks

    splits = text.split(separator)

    final_chunks = []
    small_splits = []

    for split in splits:

        if not split.strip():
            continue

        if len(split) <= chunk_size:

            small_splits.append(
                split.strip()
            )

        else:

            # Merge smaller pieces first
            if small_splits:

                final_chunks.extend(
                    merge_splits(
                        small_splits,
                        separator,
                        chunk_size,
                        chunk_overlap
                    )
                )

                small_splits = []

            # Recursively split large text
            final_chunks.extend(
                recursive_split(
                    split,
                    separators[1:],
                    chunk_size,
                    chunk_overlap
                )
            )

    # Merge remaining pieces
    if small_splits:

        final_chunks.extend(
            merge_splits(
                small_splits,
                separator,
                chunk_size,
                chunk_overlap
            )
        )

    return final_chunks

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

# 3. Token Based Chunking
def token_chunk(text, tokenizer, chunk_size = 100):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0,len(tokens), chunk_size):
        chunks_token = tokens[i: i + chunk_size]
        chunk_text= tokenizer.decode(chunks_token)
        chunks.append(chunk_text)

    return chunks
def token_count(text, tokenizer):
    return len(tokenizer.encode(text))

# 4. Token based Recursive Chunking
def merge_splits_token(
    splits,
    separator,
    chunk_size,
    tokenizer
):
    chunks = []
    current_chunk = ""

    for split in splits:

        if current_chunk:
            candidate = (
                current_chunk
                + separator
                + split
            )
        else:
            candidate = split

        if token_count(candidate, tokenizer) <= chunk_size:
            current_chunk = candidate

        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = split

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def recursive_token_chunk(
    text,
    tokenizer,
    separators=None,
    chunk_size=300,
    chunk_overlap=0
):
    if not text:
        return []

    # Check token count
    if token_count(text, tokenizer) <= chunk_size:
        return [text.strip()]

    # Default separators
    if separators is None:
        separators = [
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]

    # No separators left
    if not separators:

        tokens = tokenizer.encode(text)

        chunks = []

        step = chunk_size - chunk_overlap

        for i in range(
            0,
            len(tokens),
            step
        ):
            chunk_tokens = tokens[
                i:i + chunk_size
            ]

            chunks.append(
                tokenizer.decode(chunk_tokens)
            )

        return chunks

    separator = separators[0]

    # Special case for character/token fallback
    if separator == "":
        tokens = tokenizer.encode(text)

        chunks = []

        step = chunk_size - chunk_overlap

        for i in range(
            0,
            len(tokens),
            step
        ):
            chunk_tokens = tokens[
                i:i + chunk_size
            ]

            chunks.append(
                tokenizer.decode(chunk_tokens)
            )

        return chunks

    splits = text.split(separator)

    final_chunks = []
    small_splits = []

    for split in splits:

        split = split.strip()

        if not split:
            continue

        # Check using TOKENS
        if token_count(split, tokenizer) <= chunk_size:

            small_splits.append(split)

        else:

            # Merge previous small splits
            if small_splits:

                merged = merge_splits_token(
                    small_splits,
                    separator,
                    chunk_size,
                    tokenizer
                )

                final_chunks.extend(merged)

                small_splits = []

            # Recursive call
            smaller_chunks = recursive_token_chunk(
                text=split,
                tokenizer=tokenizer,
                separators=separators[1:],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            final_chunks.extend(smaller_chunks)

    # Merge remaining pieces
    if small_splits:

        merged = merge_splits_token(
            small_splits,
            separator,
            chunk_size,
            tokenizer
        )

        final_chunks.extend(merged)

    return final_chunks

# 5. Token based chunk overlap
def token_chunk_with_overlap(text, tokenizer, chunk_size = 100, overlap = 20):
    tokens = tokenizer.encode(text)
    chunks = []
    if(overlap >= chunk_size):
        raise ValueError("Overlap should be less than chunk_size")
    step = chunk_size - overlap
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i: i+ chunk_size]
        if not chunk_tokens:
            break
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if i + chunk_size >= len(tokens):
            break
    return chunks

# Semantic Chunking
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
def semantic_chunk(text, threshold=0.5):
    sentences = [
        sentence.strip()
        for sentence in text.split(".")
        if sentence.strip()
    ]
    if not sentences:
        return []
    embeddings = model.encode(sentences)
    similarities = []
    for i in range(len(sentences) - 1):
        similarity = cos_sim(
            embeddings[i],
            embeddings[i + 1]
        ).item()
        similarities.append(similarity)
    chunks = []
    current_chunk = [sentences[0]]
    for i, similarity in enumerate(similarities):
        next_sentence = sentences[i + 1]
        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
        else:
            current_chunk.append(next_sentence)
    chunks.append(" ".join(current_chunk))
    return chunks