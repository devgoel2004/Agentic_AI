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