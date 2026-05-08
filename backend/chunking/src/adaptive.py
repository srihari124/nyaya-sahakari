from typing import Dict


class ChunkConfig:
    """Chunk sizing settings for a document."""

    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def to_dict(self) -> Dict:
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap
        }


def get_chunk_config(text: str) -> ChunkConfig:
    """Return chunk sizing based on document length."""

    word_count = len(text.split())

    if word_count < 300:
        return ChunkConfig(
            chunk_size=None,
            overlap=0
        )

    if word_count < 1500:
        return ChunkConfig(
            chunk_size=400,
            overlap=50
        )

    if word_count < 5000:
        return ChunkConfig(
            chunk_size=600,
            overlap=75
        )

    return ChunkConfig(
        chunk_size=800,
        overlap=100
    )


def should_chunk(config: ChunkConfig) -> bool:
    """Return whether the document should be chunked."""
    return config.chunk_size is not None


def get_chunk_metadata(text: str) -> Dict:
    """Return basic size metadata for a document."""
    words = len(text.split())

    if words < 300:
        size = "small"
    elif words < 1500:
        size = "medium"
    elif words < 5000:
        size = "large"
    else:
        size = "very_large"

    return {
        "word_count": words,
        "doc_size": size
    }
