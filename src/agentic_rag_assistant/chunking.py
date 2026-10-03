"""Split extracted text into overlapping windows with source offsets."""

from dataclasses import dataclass

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class TextChunk:
    text: str
    start_char: int
    end_char: int


def chunk_text(
    text: str, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[TextChunk]:
    """Keep exact character spans; end offsets are exclusive."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]
        if window.strip():
            chunks.append(TextChunk(text=window, start_char=start, end_char=end))
        if end == len(text):
            break
        start = end - overlap
    return chunks
